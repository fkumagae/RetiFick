# RetiFick - Simulador de Retificadores Controlados

App didatico em Python/Streamlit para simular curvas de tensao e corrente em retificadores controlados.

## Topologias incluidas

- Monofasico: meia onda, meia onda com roda livre, ponto medio e Graetz.
- Trifasico: meia onda, meia onda com roda livre, ponto medio de 3 pulsos e Graetz de 6 pulsos.
- Cargas: R, RL e RLE.
- Parametros: tensao RMS, frequencia, R, L, E, alfa e beta.

## Como rodar

1. Instale o Python 3.10 ou superior.
2. No terminal, dentro da pasta do projeto, rode:

```bash
pip install -r requirements.txthttps://github.com/fkumagae/RetificAe/blob/main/README.md
streamlit run app.py
```

O navegador abrirá o app automaticamente.

## Convencao usada para beta

O beta do app e o fim da conducao medido a partir da origem natural de cada pulso.
Valores comuns para conducao continua:

- Monofasico ponto medio/Graetz: beta = alfa + 180 graus.
- Trifasico meia onda/ponto medio de 3 pulsos: beta = alfa + 120 graus.
- Trifasico Graetz de 6 pulsos: beta = alfa + 60 graus.

## Limitacoes do modelo

- Semicondutores ideais.
- Sem queda direta em SCR/diodo.
- Sem indutancia de fonte.
- Sem sobreposicao de comutacao.
- Beta e imposto pelo usuario, nao calculado automaticamente.
