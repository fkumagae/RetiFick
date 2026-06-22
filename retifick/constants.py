TOPOLOGIAS = [
    "Meia onda",
    "Meia onda com roda livre",
    "Mista",
    "Ponto medio",
    "Graetz",
]

CARGAS = ["R", "RL", "RLE"]


def topologias_por_fase(fases: str) -> list[str]:
    if fases == "Monofasico":
        return TOPOLOGIAS
    return [top for top in TOPOLOGIAS if top != "Mista"]
