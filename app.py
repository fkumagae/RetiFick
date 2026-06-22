import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


TOPOLOGIAS = [
    "Meia onda",
    "Meia onda com roda livre",
    "Ponto medio",
    "Graetz",
]

CARGAS = ["R", "RL", "RLE"]


@dataclass
class SimParams:
    fases: str
    topologia: str
    carga: str
    v_rms: float
    freq: float
    fonte_modelo: str
    r: float
    l_mH: float
    e: float
    alpha_deg: float
    beta_deg: float
    n_ciclos: int
    pontos_por_ciclo: int
    fourier_offset_pu: float = 0.0
    fourier_componentes: List[Tuple[int, float, float]] = field(default_factory=list)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def beta_ajustado(alpha_rad: float, beta_rad: float) -> float:
    """Garante que beta esteja depois de alfa dentro da referencia do pulso."""
    while beta_rad <= alpha_rad:
        beta_rad += 2.0 * np.pi
    return beta_rad


def fases_trifasicas(theta: np.ndarray, vm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    va = vm * np.sin(theta)
    vb = vm * np.sin(theta - 2.0 * np.pi / 3.0)
    vc = vm * np.sin(theta + 2.0 * np.pi / 3.0)
    return va, vb, vc


def onda_fourier(
    theta: np.ndarray,
    vm: float,
    componentes: List[Tuple[int, float, float]],
    offset_pu: float = 0.0,
) -> np.ndarray:
    sinal = np.full_like(theta, fill_value=offset_pu, dtype=float)
    for harmonica, coef_cos, coef_sin in componentes:
        sinal += coef_cos * np.cos(harmonica * theta) + coef_sin * np.sin(harmonica * theta)
    return vm * sinal


def tensoes_fonte(theta: np.ndarray, params: SimParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vm = math.sqrt(2.0) * params.v_rms
    if params.fonte_modelo == "Serie de Fourier":
        vs_mono = onda_fourier(theta, vm, params.fourier_componentes, params.fourier_offset_pu)
        va = onda_fourier(theta, vm, params.fourier_componentes, params.fourier_offset_pu)
        vb = onda_fourier(theta - 2.0 * np.pi / 3.0, vm, params.fourier_componentes, params.fourier_offset_pu)
        vc = onda_fourier(theta + 2.0 * np.pi / 3.0, vm, params.fourier_componentes, params.fourier_offset_pu)
    else:
        vs_mono = vm * np.sin(theta)
        va, vb, vc = fases_trifasicas(theta, vm)
    return vs_mono, va, vb, vc


def nome_fase(idx: int) -> str:
    return ["A", "B", "C"][idx]


def largura_natural_pulso(fases: str, topologia: str) -> float:
    """Largura natural entre disparos, em graus, para uma aproximacao didatica."""
    if fases == "Monofasico":
        if topologia == "Meia onda" or topologia == "Meia onda com roda livre":
            return 360.0
        return 180.0
    # Trifasico
    if topologia == "Graetz":
        return 60.0
    return 120.0


def construir_mascaras(params: SimParams):
    alpha = math.radians(params.alpha_deg)
    beta = beta_ajustado(alpha, math.radians(params.beta_deg))

    total_pontos = int(params.n_ciclos * params.pontos_por_ciclo) + 1
    theta = np.linspace(0.0, params.n_ciclos * 2.0 * np.pi, total_pontos)
    tempo = theta / (2.0 * np.pi * params.freq)

    vs_mono, va, vb, vc = tensoes_fonte(theta, params)
    fases_arr = [va, vb, vc]

    caminho_fonte = np.zeros_like(theta, dtype=bool)
    caminho_roda_livre = np.zeros_like(theta, dtype=bool)
    v_fonte_aplicada = np.zeros_like(theta, dtype=float)
    dispositivo = np.full_like(theta, fill_value=-1, dtype=int)

    rotulos_dispositivos: List[str] = []

    def atribuir_intervalo(
        inicio: float,
        fim: float,
        tensao_func: Callable[[np.ndarray], np.ndarray],
        disp_idx: int,
        roda_livre_inicio: Optional[float] = None,
    ) -> None:
        nonlocal caminho_fonte, caminho_roda_livre, v_fonte_aplicada, dispositivo
        if fim <= inicio:
            return

        if roda_livre_inicio is not None:
            fim_fonte = min(fim, roda_livre_inicio)
        else:
            fim_fonte = fim

        m_fonte = (theta >= inicio) & (theta < fim_fonte)
        if np.any(m_fonte):
            caminho_fonte[m_fonte] = True
            caminho_roda_livre[m_fonte] = False
            v_fonte_aplicada[m_fonte] = tensao_func(theta[m_fonte])
            dispositivo[m_fonte] = disp_idx

        if roda_livre_inicio is not None and fim > roda_livre_inicio:
            m_rl = (theta >= roda_livre_inicio) & (theta < fim)
            if np.any(m_rl):
                # A roda livre substitui a fonte; vo ideal do diodo de roda livre = 0 V.
                caminho_fonte[m_rl] = False
                caminho_roda_livre[m_rl] = True
                v_fonte_aplicada[m_rl] = 0.0
                dispositivo[m_rl] = disp_idx

    # Gera pulsos tambem antes e depois da janela para evitar bordas estranhas.
    k_min = -8
    k_max = int(params.n_ciclos * 12 + 24)

    if params.fases == "Monofasico":
        if params.topologia in ["Meia onda", "Meia onda com roda livre"]:
            rotulos_dispositivos = ["T1"]
            for k in range(k_min, k_max):
                base = 2.0 * np.pi * k
                inicio = base + alpha
                fim = base + beta
                roda_livre_inicio = None
                if params.topologia == "Meia onda com roda livre":
                    roda_livre_inicio = base + np.pi
                atribuir_intervalo(
                    inicio,
                    fim,
                    lambda th: onda_fourier(
                        th,
                        math.sqrt(2.0) * params.v_rms,
                        params.fourier_componentes,
                        params.fourier_offset_pu,
                    )
                    if params.fonte_modelo == "Serie de Fourier"
                    else math.sqrt(2.0) * params.v_rms * np.sin(th),
                    disp_idx=0,
                    roda_livre_inicio=roda_livre_inicio,
                )

        elif params.topologia in ["Ponto medio", "Graetz"]:
            rotulos_dispositivos = ["T1/T2" if params.topologia == "Graetz" else "T1",
                                    "T3/T4" if params.topologia == "Graetz" else "T2"]
            for k in range(k_min, k_max):
                base = np.pi * k
                sinal = 1.0 if (k % 2 == 0) else -1.0
                disp_idx = k % 2
                inicio = base + alpha
                fim = base + beta
                atribuir_intervalo(
                    inicio,
                    fim,
                    lambda th, s=sinal: s * (
                        onda_fourier(
                            th,
                            math.sqrt(2.0) * params.v_rms,
                            params.fourier_componentes,
                            params.fourier_offset_pu,
                        )
                        if params.fonte_modelo == "Serie de Fourier"
                        else math.sqrt(2.0) * params.v_rms * np.sin(th)
                    ),
                    disp_idx=disp_idx,
                )

    else:  # Trifasico
        if params.topologia in ["Meia onda", "Meia onda com roda livre", "Ponto medio"]:
            # No trifasico, meia onda e ponto medio representam o retificador controlado de 3 pulsos.
            rotulos_dispositivos = ["T1 fase A", "T2 fase B", "T3 fase C"]
            base0 = np.pi / 6.0  # 30 graus: inicio natural da fase A como maior fase positiva.
            for k in range(k_min, k_max):
                fase_idx = k % 3
                base = base0 + k * 2.0 * np.pi / 3.0
                inicio = base + alpha
                fim = base + beta
                roda_livre_inicio = None
                if params.topologia == "Meia onda com roda livre":
                    # A fase conduz pela fonte ate seu cruzamento por zero; depois, idealmente, a roda livre segura vo=0.
                    roda_livre_inicio = base + 5.0 * np.pi / 6.0
                atribuir_intervalo(
                    inicio,
                    fim,
                    lambda th, idx=fase_idx: onda_fourier(
                        th + [0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0][idx],
                        math.sqrt(2.0) * params.v_rms,
                        params.fourier_componentes,
                        params.fourier_offset_pu,
                    )
                    if params.fonte_modelo == "Serie de Fourier"
                    else math.sqrt(2.0) * params.v_rms * np.sin(
                        th + [0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0][idx]
                    ),
                    disp_idx=fase_idx,
                    roda_livre_inicio=roda_livre_inicio,
                )

        elif params.topologia == "Graetz":
            rotulos_dispositivos = ["T1/T6 A-B", "T1/T2 A-C", "T3/T2 B-C", "T3/T4 B-A", "T5/T4 C-A", "T5/T6 C-B"]
            pares = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
            base0 = np.pi / 6.0  # 30 graus, segmentos naturais de 60 graus.
            def tensao_linha(th: np.ndarray, p: int, n: int) -> np.ndarray:
                if params.fonte_modelo == "Serie de Fourier":
                    f = [
                        onda_fourier(th, math.sqrt(2.0) * params.v_rms, params.fourier_componentes, params.fourier_offset_pu),
                        onda_fourier(th - 2.0 * np.pi / 3.0, math.sqrt(2.0) * params.v_rms, params.fourier_componentes, params.fourier_offset_pu),
                        onda_fourier(th + 2.0 * np.pi / 3.0, math.sqrt(2.0) * params.v_rms, params.fourier_componentes, params.fourier_offset_pu),
                    ]
                else:
                    f = [
                        math.sqrt(2.0) * params.v_rms * np.sin(th),
                        math.sqrt(2.0) * params.v_rms * np.sin(th - 2.0 * np.pi / 3.0),
                        math.sqrt(2.0) * params.v_rms * np.sin(th + 2.0 * np.pi / 3.0),
                    ]
                return f[p] - f[n]

            for k in range(k_min, k_max):
                par_idx = k % 6
                p, n = pares[par_idx]
                base = base0 + k * np.pi / 3.0
                inicio = base + alpha
                fim = base + beta
                atribuir_intervalo(
                    inicio,
                    fim,
                    lambda th, pp=p, nn=n: tensao_linha(th, pp, nn),
                    disp_idx=par_idx,
                )

    return theta, tempo, vs_mono, va, vb, vc, caminho_fonte, caminho_roda_livre, v_fonte_aplicada, dispositivo, rotulos_dispositivos


def tensoes_dispositivos(
    params: SimParams,
    theta: np.ndarray,
    vs_mono: np.ndarray,
    va: np.ndarray,
    vb: np.ndarray,
    vc: np.ndarray,
    caminho_fonte: np.ndarray,
    dispositivo: np.ndarray,
    n_dispositivos: int,
) -> List[np.ndarray]:
    tensoes_base: List[np.ndarray] = []

    if params.fases == "Monofasico":
        if params.topologia in ["Meia onda", "Meia onda com roda livre"]:
            tensoes_base = [vs_mono]
        elif params.topologia in ["Ponto medio", "Graetz"]:
            tensoes_base = [vs_mono, -vs_mono]
    else:
        if params.topologia in ["Meia onda", "Meia onda com roda livre", "Ponto medio"]:
            tensoes_base = [va, vb, vc]
        elif params.topologia == "Graetz":
            tensoes_base = [va - vb, va - vc, vb - vc, vb - va, vc - va, vc - vb]

    while len(tensoes_base) < n_dispositivos:
        tensoes_base.append(np.zeros_like(theta))

    tensoes = []
    for idx in range(n_dispositivos):
        conduzindo = (dispositivo == idx) & caminho_fonte
        tensoes.append(np.where(conduzindo, 0.0, tensoes_base[idx]))

    return tensoes


def simular(params: SimParams) -> Tuple[pd.DataFrame, pd.DataFrame, str, List[str]]:
    (
        theta,
        tempo,
        vs_mono,
        va,
        vb,
        vc,
        caminho_fonte,
        caminho_roda_livre,
        v_aplicada,
        dispositivo,
        rotulos_dispositivos,
    ) = construir_mascaras(params)

    r = max(params.r, 1e-9)
    l = max(params.l_mH * 1e-3, 1e-9)
    e = params.e if params.carga == "RLE" else 0.0

    caminho = caminho_fonte | caminho_roda_livre
    v_out = np.where(caminho_fonte, v_aplicada, 0.0)

    i = np.zeros_like(theta, dtype=float)
    if params.carga == "R":
        i = np.where(caminho_fonte, np.maximum((v_out - e) / r, 0.0), 0.0)
    else:
        for k in range(1, len(theta)):
            dt = tempo[k] - tempo[k - 1]
            if caminho[k - 1]:
                u = v_out[k - 1] if caminho_fonte[k - 1] else 0.0
                di = (u - e - r * i[k - 1]) / l * dt
                i[k] = max(i[k - 1] + di, 0.0)
            else:
                i[k] = 0.0

    correntes_dispositivos = [
        np.where((dispositivo == idx) & caminho_fonte, i, 0.0)
        for idx in range(len(rotulos_dispositivos))
    ]
    tensoes_disp = tensoes_dispositivos(
        params,
        theta,
        vs_mono,
        va,
        vb,
        vc,
        caminho_fonte,
        dispositivo,
        len(rotulos_dispositivos),
    )

    df = pd.DataFrame(
        {
            "tempo_ms": tempo * 1000.0,
            "theta_graus": np.degrees(theta),
            "theta_mod_360": np.mod(np.degrees(theta), 360.0),
            "vs_mono_V": vs_mono,
            "va_V": va,
            "vb_V": vb,
            "vc_V": vc,
            "vo_V": v_out,
            "io_A": i,
            "caminho_fonte": caminho_fonte,
            "roda_livre": caminho_roda_livre,
            "dispositivo": dispositivo,
        }
    )
    for idx, corrente in enumerate(correntes_dispositivos, start=1):
        df[f"i_T{idx}_A"] = corrente
    for idx, tensao in enumerate(tensoes_disp, start=1):
        df[f"v_T{idx}_V"] = tensao

    ultimo_ciclo = df["theta_graus"] >= (params.n_ciclos - 1) * 360.0
    d = df.loc[ultimo_ciclo]
    resumo = pd.DataFrame(
        [
            ["VO medio", d["vo_V"].mean(), "V"],
            ["VO eficaz", rms(d["vo_V"].to_numpy()), "V"],
            ["IO medio", d["io_A"].mean(), "A"],
            ["IO eficaz", rms(d["io_A"].to_numpy()), "A"],
            ["P media aproximada", float(np.mean(d["vo_V"].to_numpy() * d["io_A"].to_numpy())), "W"],
            ["I T1 media", d["i_T1_A"].mean(), "A"],
            ["I T1 eficaz", rms(d["i_T1_A"].to_numpy()), "A"],
        ],
        columns=["Grandeza", "Valor", "Unidade"],
    )

    if params.fases == "Trifasico" and params.topologia == "Ponto medio":
        aviso = "No modo trifasico, 'Ponto medio' foi modelado como retificador controlado de 3 pulsos com neutro acessivel."
    elif params.topologia == "Graetz":
        aviso = "O modelo usa dispositivos ideais e comutacao instantanea. Nao inclui queda em SCR, indutancia da fonte nem sobreposicao de comutacao."
    else:
        aviso = "O modelo usa dispositivos ideais. Em cargas indutivas, beta encerra o caminho de corrente imposto pelo usuario."

    return df, resumo, aviso, rotulos_dispositivos


def main():
    st.set_page_config(page_title="Simulador de Retificadores Controlados", layout="wide")
    st.markdown(
        "<div style='text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 0.25rem;'>"
        "Eletrônica Industrial - Unesp Sorocaba - ICTS"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("Simulador de curvas de retificadores controlados")
    st.write(
        "App didatico para visualizar tensao e corrente em retificadores controlados com cargas R, RL e RLE. "
        "Os semicondutores sao ideais e o angulo beta define o fim de conducao de cada pulso."
    )

    with st.sidebar:
        st.header("Configuracao")
        fases = st.selectbox("Sistema", ["Monofasico", "Trifasico"])
        topologia = st.selectbox("Topologia", TOPOLOGIAS)
        carga = st.selectbox("Carga", CARGAS)

        st.divider()
        st.subheader("Rede")
        v_rms = st.number_input(
            "Tensao RMS de fase ou secundario [V]",
            min_value=1.0,
            max_value=10000.0,
            value=24.0,
            step=1.0,
        )
        freq = st.number_input("Frequencia [Hz]", min_value=1.0, max_value=1000.0, value=60.0, step=1.0)
        fonte_modelo = st.selectbox("Forma de onda da fonte", ["Senoidal ideal", "Serie de Fourier"])
        fourier_offset_pu = 0.0
        fourier_componentes: List[Tuple[int, float, float]] = [(1, 0.0, 1.0)]
        if fonte_modelo == "Serie de Fourier":
            fourier_offset_pu = st.number_input(
                "Termo DC a0 [pu da amplitude de pico]",
                min_value=-5.0,
                max_value=5.0,
                value=0.0,
                step=0.05,
                format="%.3f",
            )
            componentes_df = pd.DataFrame(
                [
                    {"Harmonica": 1, "a_n cosseno [pu]": 0.0, "b_n seno [pu]": 1.0},
                    {"Harmonica": 3, "a_n cosseno [pu]": 0.0, "b_n seno [pu]": 0.0},
                    {"Harmonica": 5, "a_n cosseno [pu]": 0.0, "b_n seno [pu]": 0.0},
                ]
            )
            componentes_editados = st.data_editor(
                componentes_df,
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key="fourier_componentes_editor",
            )
            fourier_componentes = []
            for _, row in componentes_editados.iterrows():
                if pd.isna(row["Harmonica"]):
                    continue
                harmonica = int(row["Harmonica"])
                if harmonica < 1:
                    continue
                coef_cos = 0.0 if pd.isna(row["a_n cosseno [pu]"]) else float(row["a_n cosseno [pu]"])
                coef_sin = 0.0 if pd.isna(row["b_n seno [pu]"]) else float(row["b_n seno [pu]"])
                if abs(coef_cos) > 1e-12 or abs(coef_sin) > 1e-12:
                    fourier_componentes.append((harmonica, coef_cos, coef_sin))
            if not fourier_componentes:
                fourier_componentes = [(1, 0.0, 1.0)]
            st.caption(
                "Serie aplicada como v(theta) = Vm x [a0 + soma(a_n cos(n theta) + b_n sin(n theta))]. "
                "Os coeficientes estao em pu da amplitude de pico Vm = raiz(2) x V_rms."
            )

        st.divider()
        st.subheader("Carga")
        r = st.number_input("R [ohm]", min_value=0.001, max_value=100000.0, value=10.0, step=1.0)
        if carga in ["RL", "RLE"]:
            l_mH = st.number_input("L [mH]", min_value=0.001, max_value=100000.0, value=50.0, step=1.0)
        else:
            l_mH = 0.001
        if carga == "RLE":
            e = st.number_input("E / forca contraeletromotriz [V]", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
        else:
            e = 0.0

        st.divider()
        st.subheader("Angulos")
        unidade_angulo = st.radio("Unidade de entrada", ["Graus", "Rad"], horizontal=True)
        largura = largura_natural_pulso(fases, topologia)
        if unidade_angulo == "Graus":
            alpha_deg = st.slider("Alfa [graus]", 0.0, 180.0, 30.0, 1.0)
            beta_sugerido = min(alpha_deg + largura, 720.0)
            beta_deg = st.slider("Beta [graus]", 0.0, 720.0, float(beta_sugerido), 1.0)
        else:
            largura_rad = math.radians(largura)
            alpha_rad = st.number_input(
                "Alfa [rad]",
                min_value=0.0,
                max_value=float(np.pi),
                value=float(np.pi / 6.0),
                step=0.01,
                format="%.4f",
            )
            beta_sugerido_rad = min(alpha_rad + largura_rad, 4.0 * np.pi)
            beta_rad = st.number_input(
                "Beta [rad]",
                min_value=0.0,
                max_value=float(4.0 * np.pi),
                value=float(beta_sugerido_rad),
                step=0.01,
                format="%.4f",
            )
            alpha_deg = math.degrees(alpha_rad)
            beta_deg = math.degrees(beta_rad)
        st.caption(
            "Use beta como fim de conducao medido a partir da origem natural do pulso. "
            "Valores continuos comuns: monofasico ponto medio/Graetz beta = alfa + 180; "
            "trifasico 3 pulsos beta = alfa + 120; trifasico Graetz beta = alfa + 60."
        )

        st.divider()
        st.subheader("Amostragem")
        n_ciclos = st.slider("Numero de ciclos", 1, 8, 3, 1)
        pontos_por_ciclo = st.slider("Pontos por ciclo", 500, 10000, 3000, 500)

    params = SimParams(
        fases=fases,
        topologia=topologia,
        carga=carga,
        v_rms=v_rms,
        freq=freq,
        fonte_modelo=fonte_modelo,
        r=r,
        l_mH=l_mH,
        e=e,
        alpha_deg=alpha_deg,
        beta_deg=beta_deg,
        n_ciclos=n_ciclos,
        pontos_por_ciclo=pontos_por_ciclo,
        fourier_offset_pu=fourier_offset_pu,
        fourier_componentes=fourier_componentes,
    )

    df, resumo, aviso, rotulos_dispositivos = simular(params)

    st.info(aviso)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sistema", fases)
    col2.metric("Topologia", topologia)
    col3.metric("Carga", carga)
    col4.metric(
        "Alfa / Beta",
        f"{alpha_deg:.1f} / {beta_deg:.1f} graus",
        f"{math.radians(alpha_deg):.4f} / {math.radians(beta_deg):.4f} rad",
    )

    rotulo_t1 = rotulos_dispositivos[0] if rotulos_dispositivos else "T1"
    rotulo_t2 = rotulos_dispositivos[1] if len(rotulos_dispositivos) > 1 else "T2"
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Saida", "Fontes", f"Tiristor {rotulo_t1}", f"Tiristor {rotulo_t2}", "Resumo"])

    with tab1:
        st.subheader("Tensao e corrente na carga")
        plot_df = df.set_index("tempo_ms")[["vo_V", "io_A"]]
        st.line_chart(plot_df, height=420)
        st.caption("Eixo x em milissegundos. vo em volts e io em amperes.")

    with tab2:
        st.subheader("Tensoes da fonte")
        if fases == "Monofasico":
            st.line_chart(df.set_index("tempo_ms")[["vs_mono_V"]], height=420)
        else:
            st.line_chart(df.set_index("tempo_ms")[["va_V", "vb_V", "vc_V"]], height=420)
        st.caption("No trifasico, a entrada informada e a tensao RMS fase-neutro.")

    with tab3:
        st.subheader(f"Sinais do primeiro tiristor ou primeiro par: {rotulo_t1}")
        modo_plot_t1 = st.radio("Plotagem", ["Corrente", "Tensao", "Ambos"], horizontal=True, key="plot_t1")
        colunas_t1 = {
            "Corrente": ["i_T1_A"],
            "Tensao": ["v_T1_V"],
            "Ambos": ["i_T1_A", "v_T1_V"],
        }[modo_plot_t1]
        st.line_chart(df.set_index("tempo_ms")[colunas_t1], height=420)
        st.caption("Quando o tiristor conduz pela fonte, a tensao ideal no dispositivo cai para aproximadamente 0 V.")

    with tab4:
        st.subheader(f"Sinais do segundo tiristor ou segundo par: {rotulo_t2}")
        if "i_T2_A" in df.columns and "v_T2_V" in df.columns:
            modo_plot_t2 = st.radio("Plotagem", ["Corrente", "Tensao", "Ambos"], horizontal=True, key="plot_t2")
            colunas_t2 = {
                "Corrente": ["i_T2_A"],
                "Tensao": ["v_T2_V"],
                "Ambos": ["i_T2_A", "v_T2_V"],
            }[modo_plot_t2]
            st.line_chart(df.set_index("tempo_ms")[colunas_t2], height=420)
            st.caption("A segunda aba acompanha o segundo tiristor ou segundo par equivalente da topologia selecionada.")
        else:
            st.info("A topologia atual possui apenas um tiristor principal monitorado.")

    with tab5:
        st.subheader("Grandezas calculadas no ultimo ciclo")
        st.dataframe(resumo, use_container_width=True, hide_index=True)
        st.subheader("Amostras")
        st.dataframe(df.tail(20), use_container_width=True, hide_index=True)

    with st.expander("Observacoes importantes do modelo"):
        st.write(
            "1. O programa e didatico: SCRs/diodos ideais, sem queda direta, sem indutancia da rede e sem overlap de comutacao.\n\n"
            "2. Em carga R, a corrente segue instantaneamente a tensao aplicada dividida por R, sempre limitada a corrente nao negativa.\n\n"
            "3. Em carga RL/RLE, a corrente e calculada numericamente por di/dt = (vo - E - R i)/L enquanto existe caminho de corrente.\n\n"
            "4. No modo com roda livre, quando a fonte deixa de alimentar a carga, o app aplica vo = 0 V e deixa a corrente decair por R, L e E.\n\n"
            "5. Beta e imposto pelo usuario. Para estudos mais avancados, beta pode ser calculado automaticamente pela extincao da corrente."
        )


if __name__ == "__main__":
    main()
