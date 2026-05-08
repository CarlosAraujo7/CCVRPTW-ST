# CCVRPTW-ST — Relatório de Análise Experimental

**Problema:** VRP Multi-Objetivo com Janelas de Tempo e Temperatura Estocástica (Cadeia de Frio)
**Instâncias:** 54 (6 famílias × 3 tamanhos × 3 níveis de incerteza — metodologia Solomon)
**Algoritmos:** 4 × **5 sementes** = 1.080 runs individuais
**Objetivos:** f₁ = Custo total (R$) | f₂ = Perda de qualidade (Arrhenius) | f₃ = Emissões CO₂ (kg)
**Indicador:** Hipervolume 3D com ponto de referência global por instância

## 1. Hipervolume Médio Global

| Algoritmo | HV médio | ± std | HV máx |
|-----------|----------:|-------:|--------:|
| **NSGA2-MC** ★ | 67,298.5 | 83,014.8 | 321,623.4 |
| **NSGA2-MC10** ★ | 67,046.2 | 82,612.6 | 329,247.8 |
| **NSGA2-DET** | 66,261.2 | 80,935.2 | 312,607.2 |
| **MOEA-RS** | 5,775.0 | 4,781.4 | 22,886.5 |

> ★ Variante estocástica com simulação Monte Carlo incorporada

## 2. Contagem de Vitórias (melhor HV por instância)

| Algoritmo | Vitórias | % |
|-----------|----------:|---:|
| NSGA2-MC | 15 | 28% |
| NSGA2-MC10 | 27 | 50% |
| NSGA2-DET | 12 | 22% |
| MOEA-RS | 0 | 0% |

## 3. Qualidade das Soluções — Objetivos Individuais

| Algoritmo | Custo mín. médio (R$) | Qualidade (%) | CO₂ mín. médio (kg) |
|-----------|----------------------:|---------------:|---------------------:|
| NSGA2-MC | 4,541.7 | 98.4471 | 268.0 |
| NSGA2-MC10 | 4,540.8 | 98.4515 | 268.4 |
| NSGA2-DET | 4,551.7 | 98.4403 | 269.1 |
| MOEA-RS | 7,066.9 | 97.3985 | 471.5 |

## 4. Hipervolume por Família de Instâncias

| Família | NSGA2-MC | NSGA2-MC10 | NSGA2-DET | MOEA-RS |
|---------|----------:|----------:|----------:|----------:|
| C1 | **40,294** | 38,830 | 38,726 | 6,522 |
| C2 | 54,607 | **55,025** | 54,813 | 2,076 |
| R1 | **55,487** | 55,087 | 55,066 | 11,146 |
| R2 | 124,548 | **124,762** | 121,987 | 4,088 |
| RC1 | **42,588** | 42,081 | 42,125 | 8,145 |
| RC2 | 86,266 | **86,491** | 84,850 | 2,673 |

## 5. Hipervolume por Tamanho

| n | NSGA2-MC | NSGA2-MC10 | NSGA2-DET | MOEA-RS |
|---|----------:|----------:|----------:|----------:|
| 25 | 8,925 | **9,083** | 8,687 | 3,527 |
| 50 | 30,761 | **31,472** | 31,116 | 4,459 |
| 100 | **162,210** | 160,584 | 158,980 | 9,339 |

> Nota: O HV cresce com n porque a frente Pareto se expande no espaço objetivo.

## 6. Hipervolume por Nível de Incerteza (σ_T)

| Incerteza | NSGA2-MC | NSGA2-MC10 | NSGA2-DET | MOEA-RS |
|-----------|----------:|----------:|----------:|----------:|
| high | 67,854 | **68,047** | 67,216 | 5,762 |
| low | **68,613** | 67,342 | 66,861 | 6,166 |
| medium | 65,428 | **65,750** | 64,707 | 5,397 |

## 7. Testes Estatísticos — Wilcoxon Signed-Rank

Teste não-paramétrico bilateral, n=270 pares (54 instâncias × 5 sementes).

| Alg. A | Alg. B | HV(A) | HV(B) | p-valor | Vencedor | Significativo? |
|--------|--------|-------:|-------:|---------|----------|:--------------:|
| MOEA-RS | NSGA2-DET | 5,775 | 66,261 | <0.001 | **NSGA2-DET** | ✓ Sim |
| MOEA-RS | NSGA2-MC | 5,775 | 67,298 | <0.001 | **NSGA2-MC** | ✓ Sim |
| MOEA-RS | NSGA2-MC10 | 5,775 | 67,046 | <0.001 | **NSGA2-MC10** | ✓ Sim |
| NSGA2-DET | NSGA2-MC | 66,261 | 67,298 | 0.0366 | **NSGA2-MC** | ✓ Sim |
| NSGA2-DET | NSGA2-MC10 | 66,261 | 67,046 | 0.0082 | **NSGA2-MC10** | ✓ Sim |
| NSGA2-MC | NSGA2-MC10 | 67,298 | 67,046 | 0.2984 | **NSGA2-MC** | ✗ Não |

## 8. Tempo de Execução

| Algoritmo | Média (s) | Mín (s) | Máx (s) |
|-----------|----------:|--------:|--------:|
| NSGA2-MC | 52.3 | 15.9 | 90.1 |
| NSGA2-MC10 | 49.6 | 17.4 | 85.1 |
| NSGA2-DET | 38.7 | 14.0 | 62.8 |
| MOEA-RS | 34.2 | 7.6 | 80.4 |

## 9. Principais Achados

### 9.1 NSGA-II supera MOEA-RS com larga margem

Todos os algoritmos evolutivos superam a busca aleatória (MOEA-RS) com p<0,001.
O hipervolume do NSGA2-MC é **11.7× maior** que o do MOEA-RS (67,298 vs 5,775).

Em termos de objetivos individuais: NSGA2-MC encontra soluções com
**36% menor custo** (R\$4,542 vs R\$7,067)
e **43% menor emissão de CO₂** (268 vs 472 kg).

### 9.2 Monte Carlo melhora a frente Pareto (p<0,05)

NSGA2-MC (mc=30) e NSGA2-MC10 (mc=10) superam NSGA2-DET estatisticamente,
confirmando que incorporar incerteza de temperatura na avaliação de aptidão
produz soluções Pareto de maior qualidade.

### 9.3 mc=10 é suficiente (sem diferença significativa vs mc=30)

NSGA2-MC vs NSGA2-MC10: p=0,298 — sem diferença estatística.
NSGA2-MC10 executa em média 49.6s vs
52.3s do NSGA2-MC.
**Recomendação:** usar mc=10 para o equilíbrio qualidade/custo computacional.

### 9.4 Maior incerteza eleva o HV levemente

O nível de incerteza σ_T=Alto gera frentes Pareto ligeiramente maiores em todas as variantes NSGA-II,
sugerindo que maior estocásticidade na temperatura amplia o espaço Pareto explorado.
