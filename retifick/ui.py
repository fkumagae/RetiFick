import math
import json
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from .config import carregar_config, indice_opcao, montar_config_persistida, salvar_config
from .constants import CARGAS, TOPOLOGIAS, topologias_por_fase
from .models import SimParams
from .simulation import largura_natural_pulso, simular


MAX_PONTOS_GRAFICO = 2000


def _componentes_fourier(config_anterior: Dict[str, Any]) -> Tuple[float, List[Tuple[int, float, float]], List[Dict[str, Any]]]:
    fourier_offset_pu = st.number_input(
        "Termo DC a0 [pu da amplitude de pico]",
        min_value=-5.0,
        max_value=5.0,
        value=float(config_anterior["fourier_offset_pu"]),
        step=0.05,
        format="%.3f",
    )
    componentes_df = pd.DataFrame(config_anterior["fourier_componentes"])
    componentes_editados = st.data_editor(
        componentes_df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="fourier_componentes_editor",
    )

    fourier_componentes: List[Tuple[int, float, float]] = []
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
    return fourier_offset_pu, fourier_componentes, componentes_editados.to_dict(orient="records")


def _rotulo_tensao_rede(fases: str, topologia: str) -> str:
    if fases == "Monofasico" and topologia == "Ponto medio":
        return "Tensao RMS de cada meia-secundaria [V]"
    if fases == "Monofasico":
        return "Tensao RMS do secundario [V]"
    return "Tensao RMS de fase [V]"


def _explicacao_tensao_rede(fases: str, topologia: str) -> str:
    if fases == "Monofasico" and topologia == "Ponto medio":
        return "No ponto medio monofasico, a entrada representa a tensao RMS de cada meia-secundaria em relacao ao tap central."
    if fases == "Monofasico" and topologia == "Mista":
        return "Na topologia mista monofasica, a entrada representa a tensao RMS do secundario da ponte semicontrolada."
    if fases == "Monofasico":
        return "No monofasico sem tap central, a entrada representa a tensao RMS do secundario aplicado a ponte ou ao retificador."
    return "No trifasico, a entrada representa a tensao RMS fase-neutro."


def _mostrar_referencias_angulares(alpha_deg: float, beta_deg: float) -> None:
    referencias = pd.DataFrame(
        [
            {"Ponto": "pi/4", "Rad": f"{math.pi / 4:.4f}", "Graus": f"{45.0:.1f}"},
            {"Ponto": "pi/2", "Rad": f"{math.pi / 2:.4f}", "Graus": f"{90.0:.1f}"},
            {"Ponto": "2pi/3", "Rad": f"{2.0 * math.pi / 3.0:.4f}", "Graus": f"{120.0:.1f}"},
            {"Ponto": "pi", "Rad": f"{math.pi:.4f}", "Graus": f"{180.0:.1f}"},
            {"Ponto": "3pi/2", "Rad": f"{3.0 * math.pi / 2.0:.4f}", "Graus": f"{270.0:.1f}"},
            {"Ponto": "2pi", "Rad": f"{2.0 * math.pi:.4f}", "Graus": f"{360.0:.1f}"},
        ]
    )
    st.caption(
        f"Alfa atual: {alpha_deg:.1f} graus ({math.radians(alpha_deg):.4f} rad). "
        f"Beta atual: {beta_deg:.1f} graus ({math.radians(beta_deg):.4f} rad)."
    )
    st.dataframe(referencias, use_container_width=True, hide_index=True)


def _serializar_params_cache(params: SimParams) -> str:
    return json.dumps(
        {
            "fases": params.fases,
            "topologia": params.topologia,
            "carga": params.carga,
            "condicao_inicial": params.condicao_inicial,
            "v_rms": params.v_rms,
            "freq": params.freq,
            "fonte_modelo": params.fonte_modelo,
            "r": params.r,
            "l_mH": params.l_mH,
            "e": params.e,
            "alpha_deg": params.alpha_deg,
            "beta_deg": params.beta_deg,
            "n_ciclos": params.n_ciclos,
            "pontos_por_ciclo": params.pontos_por_ciclo,
            "fourier_offset_pu": params.fourier_offset_pu,
            "fourier_componentes": params.fourier_componentes,
        },
        sort_keys=True,
    )


@st.cache_data(show_spinner=False)
def _simular_cached(params_json: str):
    bruto = json.loads(params_json)
    params = SimParams(
        fases=bruto["fases"],
        topologia=bruto["topologia"],
        carga=bruto["carga"],
        condicao_inicial=bruto["condicao_inicial"],
        v_rms=bruto["v_rms"],
        freq=bruto["freq"],
        fonte_modelo=bruto["fonte_modelo"],
        r=bruto["r"],
        l_mH=bruto["l_mH"],
        e=bruto["e"],
        alpha_deg=bruto["alpha_deg"],
        beta_deg=bruto["beta_deg"],
        n_ciclos=bruto["n_ciclos"],
        pontos_por_ciclo=bruto["pontos_por_ciclo"],
        fourier_offset_pu=bruto["fourier_offset_pu"],
        fourier_componentes=[tuple(item) for item in bruto["fourier_componentes"]],
    )
    return simular(params)


def _reduzir_pontos_plot(df: pd.DataFrame, max_pontos: int = MAX_PONTOS_GRAFICO) -> pd.DataFrame:
    if len(df) <= max_pontos:
        return df
    passo = max(1, len(df) // max_pontos)
    reduzido = df.iloc[::passo].copy()
    if reduzido.index[-1] != df.index[-1]:
        reduzido = pd.concat([reduzido, df.iloc[[-1]]])
    return reduzido


def _dados_referencia_angular(params: SimParams) -> pd.DataFrame:
    periodo_ms = 1000.0 / params.freq
    referencias_base = [
        ("pi/4", 45.0),
        ("pi/2", 90.0),
        ("2pi/3", 120.0),
        ("pi", 180.0),
        ("3pi/2", 270.0),
        ("2pi", 360.0),
    ]

    passo_disparo_deg = largura_natural_pulso(params.fases, params.topologia)

    def formatar_offset_pi(offset_deg: float) -> str:
        if abs(offset_deg) < 1e-9:
            return ""
        frac = Fraction(offset_deg / 180.0).limit_denominator(12)
        numerador = frac.numerator
        denominador = frac.denominator
        sinal = "+" if numerador >= 0 else "-"
        n = abs(numerador)
        if denominador == 1:
            termo = "pi" if n == 1 else f"{n}pi"
        else:
            termo = f"pi/{denominador}" if n == 1 else f"{n}pi/{denominador}"
        return f"{sinal}{termo}"

    linhas: List[Dict[str, float | str]] = []
    for ciclo in range(params.n_ciclos):
        for rotulo, angulo_deg in referencias_base:
            if rotulo == "2pi" and ciclo == params.n_ciclos - 1:
                tempo_ms = (ciclo + 1.0) * periodo_ms
            else:
                tempo_ms = (ciclo + angulo_deg / 360.0) * periodo_ms
            linhas.append(
                {
                    "tempo_ms": tempo_ms,
                    "rotulo": rotulo,
                    "angulo_deg": angulo_deg,
                }
            )

    total_janela_deg = params.n_ciclos * 360.0
    n_pulsos = int(math.ceil(total_janela_deg / passo_disparo_deg)) + 1
    for pulso in range(n_pulsos):
        origem_deg = pulso * passo_disparo_deg
        for nome_base, angulo_base in [("alfa", params.alpha_deg), ("beta", params.beta_deg)]:
            angulo_total_deg = origem_deg + angulo_base
            tempo_ms = angulo_total_deg / 360.0 * periodo_ms
            if tempo_ms < -1e-9 or tempo_ms > params.n_ciclos * periodo_ms + 1e-9:
                continue
            linhas.append(
                {
                    "tempo_ms": tempo_ms,
                    "rotulo": f"{nome_base}{formatar_offset_pi(origem_deg)}",
                    "angulo_deg": angulo_total_deg,
                }
            )
    return pd.DataFrame(linhas)


def _grafico_com_referencias(
    df_plot: pd.DataFrame,
    colunas: List[str],
    params: SimParams,
    height: int,
) -> alt.Chart:
    dados = df_plot[["tempo_ms", *colunas]].melt(
        id_vars=["tempo_ms"],
        value_vars=colunas,
        var_name="sinal",
        value_name="valor",
    )
    referencias = _dados_referencia_angular(params)

    base = alt.Chart(dados).encode(
        x=alt.X("tempo_ms:Q", title="Tempo [ms]"),
        y=alt.Y("valor:Q"),
        color=alt.Color("sinal:N", title="Sinal"),
        tooltip=[
            alt.Tooltip("tempo_ms:Q", title="Tempo [ms]", format=".3f"),
            alt.Tooltip("sinal:N", title="Sinal"),
            alt.Tooltip("valor:Q", title="Valor", format=".3f"),
        ],
    )

    linhas = base.mark_line().properties(height=height)

    regras = alt.Chart(referencias).mark_rule(strokeDash=[4, 4], opacity=0.45, color="#999999").encode(
        x="tempo_ms:Q",
        tooltip=[
            alt.Tooltip("rotulo:N", title="Referencia"),
            alt.Tooltip("angulo_deg:Q", title="Angulo [graus]", format=".1f"),
            alt.Tooltip("tempo_ms:Q", title="Tempo [ms]", format=".3f"),
        ],
    )

    textos = (
        alt.Chart(referencias)
        .mark_text(align="left", angle=270, dy=-4, dx=3, fontSize=10, color="#bbbbbb")
        .encode(
            x="tempo_ms:Q",
            y=alt.value(12),
            text="rotulo:N",
        )
    )

    return alt.layer(regras, linhas, textos).resolve_scale(y="shared")


def _bloco_tensao_media(params: SimParams) -> None:
    vm = math.sqrt(2.0) * params.v_rms
    alpha_rad = math.radians(params.alpha_deg)
    beta_rad = math.radians(params.beta_deg)

    if params.fases == "Monofasico" and params.topologia == "Meia onda":
        prefator = r"\frac{1}{2\pi}"
        pulso = r"V_m \sin(\theta)"
        fechada = rf"\overline{{V_o}} = \frac{{V_m}}{{2\pi}}\left[\cos(\alpha)-\cos(\beta)\right]"
    elif params.fases == "Monofasico" and params.topologia in ["Ponto medio", "Graetz"]:
        prefator = r"\frac{1}{\pi}"
        pulso = r"V_m \sin(\theta)"
        fechada = rf"\overline{{V_o}} = \frac{{V_m}}{{\pi}}\left[\cos(\alpha)-\cos(\beta)\right]"
    elif params.fases == "Monofasico" and params.topologia == "Mista":
        prefator = r"\frac{1}{\pi}"
        pulso = r"V_m \sin(\theta)"
        fechada = ""
    elif params.fases == "Trifasico" and params.topologia in ["Meia onda", "Meia onda com roda livre", "Ponto medio"]:
        prefator = r"\frac{3}{2\pi}"
        pulso = r"V_m \sin(\theta+\phi_k)"
        fechada = ""
    elif params.fases == "Trifasico" and params.topologia == "Graetz":
        prefator = r"\frac{3}{\pi}"
        pulso = r"v_{LL}(\theta)"
        fechada = ""
    else:
        prefator = r"\frac{1}{2\pi}"
        pulso = r"v_{pulso}(\theta)"
        fechada = ""

    with st.expander("Revelar integral da tensao media"):
        st.write("Forma geral usada para a topologia atual:")
        st.latex(rf"\overline{{V_o}} = {prefator}\int_{{\alpha}}^{{\beta}} {pulso}\, d\theta")

        if params.fonte_modelo == "Senoidal ideal" and fechada:
            st.write("Para fonte senoidal ideal, a expressao se reduz a:")
            st.latex(fechada)

        st.write("Valores atuais da simulacao:")
        st.latex(
            rf"V_m = \sqrt{{2}}\,V_{{rms}} = \sqrt{{2}}\cdot {params.v_rms:.3f} = {vm:.3f}\ \mathrm{{V}}"
        )
        st.latex(
            rf"\alpha = {params.alpha_deg:.3f}^\circ = {alpha_rad:.4f}\ \mathrm{{rad}}, \quad "
            rf"\beta = {params.beta_deg:.3f}^\circ = {beta_rad:.4f}\ \mathrm{{rad}}"
        )
        st.caption(
            "No app, beta e o fim da conducao dentro do pulso equivalente. Para topologias multipulso, o prefator ja considera a repeticao dos pulsos ao longo de 2pi."
        )


def _coletar_parametros(config_anterior: Dict[str, Any]) -> SimParams:
    with st.sidebar:
        st.header("Configuracao")
        fases_opcoes = ["Monofasico", "Trifasico"]
        condicoes_iniciais = ["Partida em t=0", "Regime permanente"]
        fontes = ["Senoidal ideal", "Serie de Fourier"]
        unidades = ["Graus", "Rad"]

        fases = st.selectbox("Sistema", fases_opcoes, index=indice_opcao(fases_opcoes, config_anterior["fases"]))
        topologias_disponiveis = topologias_por_fase(fases)
        topologia = st.selectbox(
            "Topologia",
            topologias_disponiveis,
            index=indice_opcao(topologias_disponiveis, config_anterior["topologia"]),
        )
        carga = st.selectbox("Carga", CARGAS, index=indice_opcao(CARGAS, config_anterior["carga"]))

        st.divider()
        st.subheader("Rede")
        condicao_inicial = st.selectbox(
            "Condicao inicial",
            condicoes_iniciais,
            index=indice_opcao(condicoes_iniciais, config_anterior["condicao_inicial"]),
        )
        v_rms = st.number_input(
            _rotulo_tensao_rede(fases, topologia),
            min_value=1.0,
            max_value=10000.0,
            value=float(config_anterior["v_rms"]),
            step=1.0,
        )
        st.caption(_explicacao_tensao_rede(fases, topologia))
        freq = st.number_input(
            "Frequencia [Hz]",
            min_value=1.0,
            max_value=1000.0,
            value=float(config_anterior["freq"]),
            step=1.0,
        )
        fonte_modelo = st.selectbox("Forma de onda da fonte", fontes, index=indice_opcao(fontes, config_anterior["fonte_modelo"]))

        fourier_offset_pu = 0.0
        fourier_componentes: List[Tuple[int, float, float]] = [(1, 0.0, 1.0)]
        fourier_componentes_tabela = config_anterior["fourier_componentes"]
        if fonte_modelo == "Serie de Fourier":
            fourier_offset_pu, fourier_componentes, fourier_componentes_tabela = _componentes_fourier(config_anterior)

        st.divider()
        st.subheader("Carga")
        r = st.number_input("R [ohm]", min_value=0.001, max_value=100000.0, value=float(config_anterior["r"]), step=1.0)
        if carga in ["RL", "RLE"]:
            l_mH = st.number_input(
                "L [mH]",
                min_value=0.001,
                max_value=100000.0,
                value=float(config_anterior["l_mH"]),
                step=1.0,
            )
        else:
            l_mH = 0.001
        if carga == "RLE":
            e = st.number_input(
                "E / forca contraeletromotriz [V]",
                min_value=0.0,
                max_value=10000.0,
                value=float(config_anterior["e"]),
                step=1.0,
            )
        else:
            e = 0.0

        st.divider()
        st.subheader("Angulos")
        unidade_angulo = st.radio(
            "Unidade de entrada",
            unidades,
            horizontal=True,
            index=indice_opcao(unidades, config_anterior["unidade_angulo"]),
        )
        largura = largura_natural_pulso(fases, topologia)
        if unidade_angulo == "Graus":
            alpha_deg = st.slider("Alfa [graus]", 0.0, 180.0, float(config_anterior["alpha_deg"]), 1.0)
            beta_deg = st.slider("Beta [graus]", 0.0, 720.0, min(float(config_anterior["beta_deg"]), 720.0), 1.0)
        else:
            largura_rad = math.radians(largura)
            alpha_rad = st.number_input(
                "Alfa [rad]",
                min_value=0.0,
                max_value=float(math.pi),
                value=float(config_anterior["alpha_rad"]),
                step=0.01,
                format="%.4f",
            )
            st.caption(f"Largura natural aproximada para esta topologia: {largura:.1f} graus ({largura_rad:.4f} rad).")
            beta_rad = st.number_input(
                "Beta [rad]",
                min_value=0.0,
                max_value=float(4.0 * math.pi),
                value=min(float(config_anterior["beta_rad"]), float(4.0 * math.pi)),
                step=0.01,
                format="%.4f",
            )
            alpha_deg = math.degrees(alpha_rad)
            beta_deg = math.degrees(beta_rad)

        st.caption(
            "Use beta como fim de conducao medido a partir da origem natural do pulso. "
            "Valores continuos comuns: monofasico ponto medio/Graetz beta = alfa + 180; "
            "monofasico mista: fonte ativa ate pi em cada semiciclo e roda livre depois disso; "
            "trifasico 3 pulsos beta = alfa + 120; trifasico Graetz beta = alfa + 60."
        )
        _mostrar_referencias_angulares(alpha_deg, beta_deg)

        st.divider()
        st.subheader("Amostragem")
        n_ciclos = st.slider("Numero de ciclos", 1, 8, int(config_anterior["n_ciclos"]), 1)
        pontos_por_ciclo = st.slider("Pontos por ciclo", 500, 10000, int(config_anterior["pontos_por_ciclo"]), 500)

    salvar_config(
        montar_config_persistida(
            config_anterior=config_anterior,
            fases=fases,
            topologia=topologia,
            carga=carga,
            condicao_inicial=condicao_inicial,
            v_rms=v_rms,
            freq=freq,
            fonte_modelo=fonte_modelo,
            r=r,
            l_mH=l_mH,
            e=e,
            unidade_angulo=unidade_angulo,
            alpha_deg=alpha_deg,
            beta_deg=beta_deg,
            n_ciclos=n_ciclos,
            pontos_por_ciclo=pontos_por_ciclo,
            fourier_offset_pu=fourier_offset_pu,
            fourier_componentes_tabela=fourier_componentes_tabela,
        )
    )

    return SimParams(
        fases=fases,
        topologia=topologia,
        carga=carga,
        condicao_inicial=condicao_inicial,
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


def main() -> None:
    config_anterior = carregar_config()

    st.set_page_config(page_title="RetiFick - Simulador de Retificadores Controlados", layout="wide")
    st.markdown(
        "<div style='text-align: center; font-size: 0.85rem; color: #666; margin-bottom: 0.25rem;'>"
        "Eletrônica Industrial - Unesp Sorocaba - ICTS "
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("RetiFick - Simulador de curvas de retificadores controlados")
    st.write(
        "App didatico para visualizar tensao e corrente em retificadores controlados com cargas R, RL e RLE. "
        "Os semicondutores sao ideais, o angulo beta define o fim de conducao de cada pulso "
        "e a simulacao pode representar partida em t=0 ou regime permanente."
    )

    params = _coletar_parametros(config_anterior)
    total_pontos = int(params.n_ciclos * params.pontos_por_ciclo) + 1
    if total_pontos > 12000:
        st.warning(
            f"A simulacao atual gera {total_pontos} amostras. Os graficos vao usar uma versao reduzida para manter a pagina responsiva."
        )

    df, resumo, aviso, rotulos_dispositivos = _simular_cached(_serializar_params_cache(params))
    plot_df = _reduzir_pontos_plot(df)

    st.info(aviso)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sistema", params.fases)
    col2.metric("Topologia", params.topologia)
    col3.metric("Carga", params.carga)
    col4.metric(
        "Alfa / Beta",
        f"{params.alpha_deg:.1f} / {params.beta_deg:.1f} graus",
        f"{math.radians(params.alpha_deg):.4f} / {math.radians(params.beta_deg):.4f} rad",
    )
    st.caption(f"Condicao inicial ativa: {params.condicao_inicial}.")
    st.caption(_explicacao_tensao_rede(params.fases, params.topologia))

    rotulo_t1 = rotulos_dispositivos[0] if rotulos_dispositivos else "T1"
    rotulo_t2 = rotulos_dispositivos[1] if len(rotulos_dispositivos) > 1 else "T2"
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Saida", "Fontes", f"Tiristor {rotulo_t1}", f"Tiristor {rotulo_t2}", "Resumo"])

    with tab1:
        st.subheader("Tensao na carga")
        st.altair_chart(_grafico_com_referencias(plot_df, ["vo_V"], params, 300), use_container_width=True)
        st.caption("Eixo x em milissegundos. vo em volts.")
        _bloco_tensao_media(params)

        st.subheader("Corrente na carga")
        st.altair_chart(_grafico_com_referencias(plot_df, ["io_A"], params, 300), use_container_width=True)
        st.caption("io usa um eixo proprio para nao parecer zerada quando comparada com vo, que costuma ter amplitude muito maior.")

    with tab2:
        st.subheader("Tensoes da fonte")
        if params.fases == "Monofasico":
            st.altair_chart(_grafico_com_referencias(plot_df, ["vs_mono_V"], params, 420), use_container_width=True)
        else:
            st.altair_chart(_grafico_com_referencias(plot_df, ["va_V", "vb_V", "vc_V"], params, 420), use_container_width=True)
        st.caption("No trifasico, a entrada informada e a tensao RMS fase-neutro.")

    with tab3:
        st.subheader(f"Sinais do primeiro tiristor ou primeiro par: {rotulo_t1}")
        modo_plot_t1 = st.radio("Plotagem", ["Corrente", "Tensao", "Ambos"], horizontal=True, key="plot_t1")
        colunas_t1 = {
            "Corrente": ["i_T1_A"],
            "Tensao": ["v_T1_V"],
            "Ambos": ["i_T1_A", "v_T1_V"],
        }[modo_plot_t1]
        st.altair_chart(_grafico_com_referencias(plot_df, colunas_t1, params, 420), use_container_width=True)
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
            st.altair_chart(_grafico_com_referencias(plot_df, colunas_t2, params, 420), use_container_width=True)
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
            "5. Em 'Partida em t=0', o primeiro ciclo nao herda conducao anterior; em 'Regime permanente', a simulacao admite pulsos antes de t = 0 para reduzir efeitos de borda.\n\n"
            "6. Beta e imposto pelo usuario. Para estudos mais avancados, beta pode ser calculado automaticamente pela extincao da corrente."
        )
