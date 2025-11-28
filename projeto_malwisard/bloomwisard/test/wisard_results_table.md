## Tabela de Resultados - BloomWiSARD (Malevis)

| Arquivo                              | tuple_size | bloom_size | Treino (s) | Avaliação (s) | Acertos | Total | Acurácia (%) |
|--------------------------------------|------------|------------|------------|---------------|---------|-------|--------------|
| wisard_results_20251127_225958.txt   | 16         | 2000       | 87.80      | 108.04        | 4145    | 5126  | 80.86        |
| wisard_results_20251127_225630.txt   | 14         | 2000       | 32.90      | 118.41        | 4135    | 5126  | 80.67        |
| wisard_results_20251127_224658.txt   | 13         | 3000       | 36.42      | 129.89        | 4128    | 5126  | 80.53        |
| wisard_results_20251127_230441.txt   | 12         | 1000       | 30.14      | 92.89         | 4114    | 5126  | 80.26        |
| wisard_results_20251127_225120.txt   | 12         | 4000       | 37.12      | 136.49        | 4113    | 5126  | 80.24        |
| wisard_results_20251127_223859.txt   | 12         | 2000       | 101.84     | 137.14        | 4111    | 5126  | 80.20        |
| wisard_results_20251127_221901.txt   | 10         | 2000       | 38.70      | 160.90        | 3971    | 5126  | 77.47        |
| wisard_results_20251127_223321.txt   | 10         | 4000       | 40.92      | 163.64        | 3964    | 5126  | 77.33        |
| wisard_results_20251127_202629.txt   | 8          | 1000       | 34.15      | 96.10         | 3326    | 5126  | 64.88        |

Observação: valores extraídos automaticamente das linhas “Parameters”, “Training time”, “Evaluation time” e “Accuracy” de cada arquivo .txt.

### Interpretação
- **Melhor tuple_size**: 16 — obteve a maior acurácia (80.86%). Há um ganho consistente ao aumentar de 12 para 14 para 16. Custo: o tempo de treino cresce (87.80 s em tuple_size=16, acima das demais execuções).
- **Melhor bloom_size**: 2000 — o melhor resultado usa 2000; com tuple_size=10, 2000 superou 4000 (77.47% vs 77.33%); com tuple_size=12, 1000/2000/4000 ficaram muito próximos (80.26/80.20/80.24). Dentro deste recorte, 4000 não trouxe ganho.
- **Recomendação**:
  - Se prioriza acurácia: usar `tuple_size=16` e `bloom_size=2000`.
  - Se prioriza tempo/recursos: `tuple_size=12` e `bloom_size=1000` mantém ~80.26% com treino/avaliação mais rápidos.
  - Passos futuros: varrer pontos intermediários (ex.: `tuple_size` 15/17 e `bloom_size` 1500/2500) e repetir execuções para checar variância.


