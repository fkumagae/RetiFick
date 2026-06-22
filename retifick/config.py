import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


CONFIG_PATH = Path(".retifick_config.json")


def config_padrao() -> Dict[str, Any]:
    return {
        "fases": "Monofasico",
        "topologia": "Meia onda",
        "carga": "R",
        "condicao_inicial": "Partida em t=0",
        "v_rms": 220.0,
        "freq": 60.0,
        "fonte_modelo": "Senoidal ideal",
        "r": 10.0,
        "l_mH": 50.0,
        "e": 0.0,
        "unidade_angulo": "Graus",
        "alpha_deg": 30.0,
        "beta_deg": 390.0,
        "alpha_rad": float(np.pi / 6.0),
        "beta_rad": float(13.0 * np.pi / 6.0),
        "n_ciclos": 3,
        "pontos_por_ciclo": 3000,
        "fourier_offset_pu": 0.0,
        "fourier_componentes": [
            {"Harmonica": 1, "a_n cosseno [pu]": 0.0, "b_n seno [pu]": 1.0},
            {"Harmonica": 3, "a_n cosseno [pu]": 0.0, "b_n seno [pu]": 0.0},
            {"Harmonica": 5, "a_n cosseno [pu]": 0.0, "b_n seno [pu]": 0.0},
        ],
    }


def carregar_config() -> Dict[str, Any]:
    config = config_padrao()
    if not CONFIG_PATH.exists():
        return config
    try:
        carregado = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return config
    if isinstance(carregado, dict):
        config.update(carregado)
    return config


def salvar_config(config: Dict[str, Any]) -> None:
    try:
        conteudo_novo = json.dumps(config, ensure_ascii=True, indent=2)
    except (TypeError, ValueError):
        return

    try:
        if CONFIG_PATH.exists():
            conteudo_atual = CONFIG_PATH.read_text(encoding="utf-8")
            if conteudo_atual == conteudo_novo:
                return
        CONFIG_PATH.write_text(conteudo_novo, encoding="utf-8")
    except OSError:
        pass


def indice_opcao(opcoes: List[str], valor: str) -> int:
    return opcoes.index(valor) if valor in opcoes else 0


def montar_config_persistida(
    *,
    config_anterior: Dict[str, Any],
    fases: str,
    topologia: str,
    carga: str,
    condicao_inicial: str,
    v_rms: float,
    freq: float,
    fonte_modelo: str,
    r: float,
    l_mH: float,
    e: float,
    unidade_angulo: str,
    alpha_deg: float,
    beta_deg: float,
    n_ciclos: int,
    pontos_por_ciclo: int,
    fourier_offset_pu: float,
    fourier_componentes_tabela: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "fases": fases,
        "topologia": topologia,
        "carga": carga,
        "condicao_inicial": condicao_inicial,
        "v_rms": float(v_rms),
        "freq": float(freq),
        "fonte_modelo": fonte_modelo,
        "r": float(r),
        "l_mH": float(l_mH),
        "e": float(e),
        "unidade_angulo": unidade_angulo,
        "alpha_deg": float(alpha_deg),
        "beta_deg": float(beta_deg),
        "alpha_rad": float(math.radians(alpha_deg)),
        "beta_rad": float(math.radians(beta_deg)),
        "n_ciclos": int(n_ciclos),
        "pontos_por_ciclo": int(pontos_por_ciclo),
        "fourier_offset_pu": float(fourier_offset_pu),
        "fourier_componentes": (
            fourier_componentes_tabela
            if fonte_modelo == "Serie de Fourier"
            else config_anterior["fourier_componentes"]
        ),
    }
