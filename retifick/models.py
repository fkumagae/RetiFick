from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimParams:
    fases: str
    topologia: str
    carga: str
    condicao_inicial: str
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
