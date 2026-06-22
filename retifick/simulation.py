import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import SimParams


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def beta_ajustado(alpha_rad: float, beta_rad: float) -> float:
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


def largura_natural_pulso(fases: str, topologia: str) -> float:
    if fases == "Monofasico":
        if topologia in ["Meia onda", "Meia onda com roda livre"]:
            return 360.0
        return 180.0
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

        fim_fonte = min(fim, roda_livre_inicio) if roda_livre_inicio is not None else fim

        m_fonte = (theta >= inicio) & (theta < fim_fonte)
        if np.any(m_fonte):
            caminho_fonte[m_fonte] = True
            caminho_roda_livre[m_fonte] = False
            v_fonte_aplicada[m_fonte] = tensao_func(theta[m_fonte])
            dispositivo[m_fonte] = disp_idx

        if roda_livre_inicio is not None and fim > roda_livre_inicio:
            m_rl = (theta >= roda_livre_inicio) & (theta < fim)
            if np.any(m_rl):
                caminho_fonte[m_rl] = False
                caminho_roda_livre[m_rl] = True
                v_fonte_aplicada[m_rl] = 0.0
                dispositivo[m_rl] = disp_idx

    k_min = 0 if params.condicao_inicial == "Partida em t=0" else -8
    k_max = int(params.n_ciclos * 12 + 24)

    if params.fases == "Monofasico":
        if params.topologia in ["Meia onda", "Meia onda com roda livre"]:
            rotulos_dispositivos = ["T1"]
            for k in range(k_min, k_max):
                base = 2.0 * np.pi * k
                inicio = base + alpha
                fim = base + beta
                roda_livre_inicio = base + np.pi if params.topologia == "Meia onda com roda livre" else None
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
        elif params.topologia == "Mista":
            rotulos_dispositivos = ["T1/D2", "T2/D1"]
            for k in range(k_min, k_max):
                base = np.pi * k
                sinal = 1.0 if (k % 2 == 0) else -1.0
                disp_idx = k % 2
                inicio = base + alpha
                fim = base + beta
                # Na topologia mista monofasica, a fonte alimenta a carga ate o fim do semiciclo.
                # Depois disso, a corrente remanescente circula em roda livre com vo = 0.
                roda_livre_inicio = base + np.pi
                atribuir_intervalo(
                    inicio,
                    fim,
                    lambda th, s=sinal: s
                    * (
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
                    roda_livre_inicio=roda_livre_inicio,
                )
        elif params.topologia in ["Ponto medio", "Graetz"]:
            rotulos_dispositivos = [
                "T1/T2" if params.topologia == "Graetz" else "T1",
                "T3/T4" if params.topologia == "Graetz" else "T2",
            ]
            for k in range(k_min, k_max):
                base = np.pi * k
                sinal = 1.0 if (k % 2 == 0) else -1.0
                disp_idx = k % 2
                inicio = base + alpha
                fim = base + beta
                atribuir_intervalo(
                    inicio,
                    fim,
                    lambda th, s=sinal: s
                    * (
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
    else:
        if params.topologia in ["Meia onda", "Meia onda com roda livre", "Ponto medio"]:
            rotulos_dispositivos = ["T1 fase A", "T2 fase B", "T3 fase C"]
            base0 = np.pi / 6.0
            for k in range(k_min, k_max):
                fase_idx = k % 3
                base = base0 + k * 2.0 * np.pi / 3.0
                inicio = base + alpha
                fim = base + beta
                roda_livre_inicio = base + 5.0 * np.pi / 6.0 if params.topologia == "Meia onda com roda livre" else None
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
            base0 = np.pi / 6.0

            def tensao_linha(th: np.ndarray, p: int, n: int) -> np.ndarray:
                if params.fonte_modelo == "Serie de Fourier":
                    fases = [
                        onda_fourier(th, math.sqrt(2.0) * params.v_rms, params.fourier_componentes, params.fourier_offset_pu),
                        onda_fourier(
                            th - 2.0 * np.pi / 3.0,
                            math.sqrt(2.0) * params.v_rms,
                            params.fourier_componentes,
                            params.fourier_offset_pu,
                        ),
                        onda_fourier(
                            th + 2.0 * np.pi / 3.0,
                            math.sqrt(2.0) * params.v_rms,
                            params.fourier_componentes,
                            params.fourier_offset_pu,
                        ),
                    ]
                else:
                    fases = [
                        math.sqrt(2.0) * params.v_rms * np.sin(th),
                        math.sqrt(2.0) * params.v_rms * np.sin(th - 2.0 * np.pi / 3.0),
                        math.sqrt(2.0) * params.v_rms * np.sin(th + 2.0 * np.pi / 3.0),
                    ]
                return fases[p] - fases[n]

            for k in range(k_min, k_max):
                par_idx = k % 6
                p, n = pares[par_idx]
                base = base0 + k * np.pi / 3.0
                inicio = base + alpha
                fim = base + beta
                atribuir_intervalo(inicio, fim, lambda th, pp=p, nn=n: tensao_linha(th, pp, nn), disp_idx=par_idx)

    return theta, tempo, vs_mono, va, vb, vc, caminho_fonte, caminho_roda_livre, v_fonte_aplicada, dispositivo, rotulos_dispositivos


def tensoes_dispositivos(
    params: SimParams,
    theta: np.ndarray,
    vs_mono: np.ndarray,
    va: np.ndarray,
    vb: np.ndarray,
    vc: np.ndarray,
    v_out: np.ndarray,
    caminho_fonte: np.ndarray,
    dispositivo: np.ndarray,
    n_dispositivos: int,
) -> List[np.ndarray]:
    if params.fases == "Monofasico":
        if params.topologia in ["Meia onda", "Meia onda com roda livre"]:
            tensoes_base: List[np.ndarray] = [vs_mono]
        elif params.topologia == "Mista":
            tensoes_base = [vs_mono, -vs_mono]
        elif params.topologia == "Ponto medio":
            anodos = [vs_mono, -vs_mono]
            tensoes = []
            for idx, anodo in enumerate(anodos):
                conduzindo = (dispositivo == idx) & caminho_fonte
                tensoes.append(np.where(conduzindo, 0.0, anodo - v_out))
            return tensoes
        else:
            tensoes_base = [vs_mono, -vs_mono]
    else:
        if params.topologia in ["Meia onda", "Meia onda com roda livre", "Ponto medio"]:
            anodos = [va, vb, vc]
            tensoes = []
            for idx, anodo in enumerate(anodos):
                conduzindo = (dispositivo == idx) & caminho_fonte
                tensoes.append(np.where(conduzindo, 0.0, anodo - v_out))
            return tensoes
        else:
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
        v_out,
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
    elif params.topologia == "Mista":
        aviso = "Na topologia mista monofasica, a carga recebe pulsos positivos da fonte e vo = 0 nos intervalos de roda livre. Nao ha tensao negativa ideal na carga."
    else:
        aviso = "O modelo usa dispositivos ideais. Em cargas indutivas, beta encerra o caminho de corrente imposto pelo usuario."

    return df, resumo, aviso, rotulos_dispositivos
