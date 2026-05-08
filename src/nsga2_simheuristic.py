"""
NSGA-II com Simulação Monte Carlo — CCVRPTW-ST
===============================================
Simheurística multiobjetivo para distribuição em cadeia de frio
sob incerteza estocástica de temperatura.

Objetivos (todos minimizados):
  f1 — Custo total de transporte (R$): custo fixo por veículo + variável por km
  f2 — Perda de qualidade (adimensional, 0–1): 1 - qualidade residual média
       ponderada por valor da carga, avaliada via Monte Carlo
  f3 — Emissões de CO₂ (kg CO₂-eq): motor + sistema de refrigeração

Algoritmo:
  1. Inicialização: heurística de vizinho mais próximo com perturbação
     aleatória (biased-randomized) + permutações aleatórias puras
  2. Avaliação: Monte Carlo embarcado (MC_SAMPLES replicações de temperatura)
  3. NSGA-II: classificação não-dominada rápida + distância de aglomeração
     + seleção por torneio binário + crossover OX + mutação 2-opt / or-opt
  4. Métricas: Hipervolume (HV), Generational Distance (GD), Spread (Δ)

Representação da solução:
  Giant-tour — permutação de {1, …, n} decodificada em rotas por
  inserção sequencial respeitando capacidade e janelas de tempo.

Referências:
  [1] Deb et al. (2002) IEEE Trans. Evol. Comput. 6(2):182–197.
  [2] Juan et al. (2021) Operations Research Perspectives 8:100124.
  [3] Ndraha et al. (2018) Food Control 89:12–21.
  [4] Solomon (1987) Oper. Res. 35(2):254–265.
  [5] Zitzler & Thiele (1999) Evol. Comput. 7(2):173–195.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. PARÂMETROS DO ALGORITMO
# ─────────────────────────────────────────────────────────────────────────────

POP_SIZE         = 100      # tamanho da população
N_GEN            = 200      # número de gerações
CROSSOVER_RATE   = 0.90     # probabilidade de crossover (OX)
MUTATION_RATE    = 0.30     # probabilidade de mutação (2-opt / or-opt)
MC_SAMPLES       = 30       # replicações Monte Carlo por avaliação
TOURNAMENT_K     = 2        # tamanho do torneio
PENALTY_INFEAS   = 1e9      # penalidade de infeasibilidade
EXTRA_VEHICLE_COST = 400.0  # custo de locacao por veiculo adicional (R$)
                            # > 2x custo fixo, incentiva uso do frota nominal
R_GAS            = 8.314    # constante universal dos gases (J mol⁻¹ K⁻¹)


# ─────────────────────────────────────────────────────────────────────────────
# 2. REPRESENTAÇÃO DE SOLUÇÃO
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Individual:
    """
    Indivíduo do NSGA-II.

    Cromossomo: giant-tour — permutação dos IDs de clientes (1-indexados).
    Decodificado em rotas por inserção sequencial com cortes baseados em
    capacidade e janelas de tempo.
    """
    chromosome: List[int]
    routes:     List[List[int]] = field(default_factory=list)
    f_cost:     float = float('inf')
    f_qloss:    float = float('inf')
    f_co2:      float = float('inf')
    rank:       int   = 0
    crowding:   float = 0.0
    feasible:   bool  = True

    def obj(self) -> Tuple[float, float, float]:
        return (self.f_cost, self.f_qloss, self.f_co2)

    def dominates(self, other: Individual) -> bool:
        """True se self domina other em todos os objetivos (pelo menos um estrito)."""
        a, b = self.obj(), other.obj()
        return (all(ai <= bi for ai, bi in zip(a, b))
                and any(ai < bi for ai, bi in zip(a, b)))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARREGAMENTO DA INSTÂNCIA E MATRIZES
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def build_matrices(inst: dict) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Constrói matrizes de distância euclidiana (km) e tempo de percurso (min).
    Índice 0 = depósito; índices 1..n = clientes.
    """
    nodes = [inst['depot']] + inst['customers']
    n     = len(nodes)
    speed = inst['fleet']['speed_km_h']
    dist  = [[0.0] * n for _ in range(n)]
    ttime = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = nodes[i]['x'], nodes[i]['y']
        for j in range(n):
            if i != j:
                dx = xi - nodes[j]['x']
                dy = yi - nodes[j]['y']
                d  = math.hypot(dx, dy)
                dist[i][j]  = round(d, 4)
                ttime[i][j] = round(d / speed * 60.0, 4)
    return dist, ttime


# ─────────────────────────────────────────────────────────────────────────────
# 4. DECODIFICADOR GIANT-TOUR → ROTAS
# ─────────────────────────────────────────────────────────────────────────────

def decode(
    chromosome: List[int],
    inst: dict,
    ttime: List[List[float]],
) -> Tuple[List[List[int]], bool]:
    """
    Decoder paralelo (parallel-route insertion) para giant-tour CVRPTW.

    Para cada cliente no cromossomo, tenta inserir em QUALQUER rota aberta
    (first-fit: primeira que respeite capacidade e janela de tempo).
    Só abre nova rota se nenhuma existente aceitar o cliente.
    Se o cliente não puder ser servido nem com veículo dedicado (chega após
    due_time mesmo saindo direto do depósito), marca infeasível.

    Esta estratégia é muito mais eficiente que o decoder sequencial (single-route)
    e produz soluções com número de rotas próximo ao ótimo greedy EDD.

    Referência: Nagata & Bräysy (2009) Networks 54(4):232-245.
    """
    customers = inst['customers']
    capacity  = inst['fleet']['capacity_kg']

    # Estado de cada rota aberta: (load, current_time, last_node)
    r_load: List[float] = []
    r_time: List[float] = []
    r_last: List[int]   = []
    routes: List[List[int]] = []
    feasible = True

    for cid in chromosome:
        c   = customers[cid - 1]
        dem = c['demand_kg']
        rt  = c['ready_time_min']
        due = c['due_date_min']
        st  = c['service_time_min']

        inserted = False
        for k in range(len(routes)):
            if r_load[k] + dem > capacity:
                continue
            travel  = ttime[r_last[k]][cid]
            arrival = r_time[k] + travel
            if arrival > due:
                continue
            wait       = max(0.0, rt - arrival)
            r_load[k] += dem
            r_time[k]  = arrival + wait + st
            r_last[k]  = cid
            routes[k].append(cid)
            inserted = True
            break

        if not inserted:
            # Abre nova rota saindo do depósito
            arr0 = ttime[0][cid]
            if arr0 > due:
                feasible = False
                continue          # cliente não atendível (due_time muito curto)
            wait = max(0.0, rt - arr0)
            r_load.append(dem)
            r_time.append(arr0 + wait + st)
            r_last.append(cid)
            routes.append([cid])

    return routes, feasible


# ─────────────────────────────────────────────────────────────────────────────
# 5. AVALIAÇÃO DE OBJETIVOS
# ─────────────────────────────────────────────────────────────────────────────

def _route_quality_mc(
    route: List[int],
    inst:  dict,
    ttime: List[List[float]],
    rng:   np.random.Generator,
    mc_samples: int,
) -> float:
    """
    Qualidade residual média ponderada por valor para uma rota, via Monte Carlo.

    Modelo de degradação de Arrhenius de primeira ordem:
      Q(t) = exp(-k(T) · t),   k(T) = A₀ · exp(-Eₐ / (R · T_K))

    Temperatura em cada trecho: T_nominal + ε, ε ~ N(0, σ_T²),
    grampeada em [T_min, T_max] para realismo físico.

    Retorna: E[Σᵢ valueᵢ · Qᵢ] / Σᵢ valueᵢ   (escalar em [0, 1]).
    """
    if not route:
        return 1.0

    customers = inst['customers']
    tp        = inst['thermal_profile']
    sigma_T   = tp['sigma_T_C']
    T_set     = tp['T_set_C']
    T_min     = tp.get('T_min_C', T_set - 3.0)
    T_max     = tp.get('T_max_C', T_set + 5.0)
    Ea        = tp['Ea_J_mol']
    A0        = tp['A0']

    # Sequência de nós: depósito (0) → c1 → c2 → ... → ck
    nodes     = [0] + route
    n_legs    = len(nodes) - 1

    # Tempos de percurso por trecho (minutos → horas)
    leg_t_h   = np.array([ttime[nodes[i]][nodes[i + 1]] / 60.0
                           for i in range(n_legs)])

    # Pesos de valor (para média ponderada)
    values    = np.array([customers[cid - 1].get('load_value_BRL', 1.0)
                           for cid in route])
    total_val = values.sum() or 1.0

    # Modo determinístico (mc_samples=0): temperatura fixa em T_set
    if mc_samples == 0:
        T_K_det = np.full(n_legs, T_set + 273.15)
        k_det   = A0 * np.exp(-Ea / (R_GAS * T_K_det))
        cum_kdt = np.cumsum(k_det * leg_t_h)
        Q_det   = np.exp(-cum_kdt)
        return float((Q_det * values).sum() / total_val)

    # Monte Carlo vetorizado — shape (mc_samples, n_legs)
    eps   = rng.normal(0.0, sigma_T, size=(mc_samples, n_legs))
    T_mat = np.clip(T_set + eps, T_min, T_max)   # temperaturas por trecho

    # Taxa de degradação k(T) por trecho (vetorizado)
    T_K   = T_mat + 273.15
    k_mat = A0 * np.exp(-Ea / (R_GAS * T_K))     # (mc_samples, n_legs)

    # Degradação acumulada: Q_i = exp(-Σ_{j≤i} k_j · t_j)
    kdt         = k_mat * leg_t_h                 # (mc_samples, n_legs)
    cum_kdt     = np.cumsum(kdt, axis=1)          # degradação até cliente i
    Q_mat       = np.exp(-cum_kdt)                # qualidade por cliente

    # Qualidade ponderada por replicação e média
    wq_per_rep = (Q_mat * values).sum(axis=1) / total_val   # (mc_samples,)
    return float(wq_per_rep.mean())


def evaluate(
    ind:        Individual,
    inst:       dict,
    dist:       List[List[float]],
    ttime:      List[List[float]],
    rng:        np.random.Generator,
    mc_samples: int = MC_SAMPLES,
):
    """
    Avalia os três objetivos de um indivíduo:
      f1 = custo total (R$)
      f2 = perda de qualidade ponderada por valor (Monte Carlo)
      f3 = emissões de CO₂ (kg CO₂-eq)

    Infeasibilidade é penalizada adicionando PENALTY_INFEAS a f1 e f3.
    """
    routes, feasible = decode(ind.chromosome, inst, ttime)
    ind.routes   = routes
    ind.feasible = feasible

    if not routes:
        ind.f_cost  = PENALTY_INFEAS
        ind.f_qloss = 1.0
        ind.f_co2   = PENALTY_INFEAS
        return

    fleet    = inst['fleet']
    co2p     = inst.get('co2_params', {})
    fix_cost = fleet['fixed_cost_BRL']
    var_cost = fleet['variable_cost_BRL_km']
    speed    = fleet['speed_km_h']
    alpha    = co2p.get('alpha_kg_per_km', 0.21)
    beta     = co2p.get('beta_kg_per_kWh', 0.092)
    P_ref    = co2p.get('P_refrigeration_kW', 3.5)

    customers  = inst['customers']
    total_cost = 0.0
    total_co2  = 0.0
    total_wq   = 0.0
    total_val  = 0.0

    for route in routes:
        nodes  = [0] + route + [0]
        r_dist = sum(dist[nodes[i]][nodes[i + 1]] for i in range(len(nodes) - 1))
        r_time_h = r_dist / speed

        total_cost += fix_cost + var_cost * r_dist
        total_co2  += alpha * r_dist + beta * P_ref * r_time_h

        r_wq  = _route_quality_mc(route, inst, ttime, rng, mc_samples)
        r_val = sum(customers[cid - 1].get('load_value_BRL', 1.0) for cid in route)
        total_wq  += r_wq * r_val
        total_val += r_val

    wq_global = total_wq / total_val if total_val > 0 else 0.0
    pen = PENALTY_INFEAS if not feasible else 0.0

    # Custo de veiculos extras alem da frota nominal
    n_nominal   = inst['fleet']['num_vehicles']
    n_extra     = max(0, len(routes) - n_nominal)
    extra_cost  = n_extra * EXTRA_VEHICLE_COST

    ind.f_cost  = total_cost + extra_cost + pen
    ind.f_qloss = 1.0 - wq_global
    ind.f_co2   = total_co2 + pen


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPERADORES DO NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

def fast_nondominated_sort(pop: List[Individual]) -> List[List[Individual]]:
    """
    Classificação rápida não-dominada — Deb et al. (2002), Seção III-A.
    Complexidade O(M·N²) onde M = número de objetivos, N = tamanho da pop.
    Retorna lista de fronts (front[0] é a Frente de Pareto).
    """
    n     = len(pop)
    S     = [[] for _ in range(n)]   # S[i]: conjunto dominado por i
    n_dom = [0] * n                  # n_dom[i]: quantos dominam i
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if pop[i].dominates(pop[j]):
                S[i].append(j)
                n_dom[j] += 1
            elif pop[j].dominates(pop[i]):
                S[j].append(i)
                n_dom[i] += 1
        if n_dom[i] == 0:
            pop[i].rank = 1
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    pop[j].rank = k + 2
                    nxt.append(j)
        fronts.append(nxt)
        k += 1

    return [[pop[i] for i in front] for front in fronts if front]


def crowding_distance_assign(front: List[Individual]):
    """
    Atribui distância de aglomeração (crowding distance) aos indivíduos de um front.
    Soluções extremas recebem distância infinita.
    """
    n = len(front)
    for ind in front:
        ind.crowding = 0.0
    if n <= 2:
        for ind in front:
            ind.crowding = float('inf')
        return

    for get_f in (lambda x: x.f_cost, lambda x: x.f_qloss, lambda x: x.f_co2):
        sf      = sorted(front, key=get_f)
        sf[0].crowding  = float('inf')
        sf[-1].crowding = float('inf')
        f_range = get_f(sf[-1]) - get_f(sf[0])
        if f_range == 0:
            continue
        for i in range(1, n - 1):
            sf[i].crowding += (get_f(sf[i + 1]) - get_f(sf[i - 1])) / f_range


def tournament_select(pop: List[Individual], rng_py: random.Random) -> Individual:
    """Seleção por torneio binário (rank menor → melhor; empate → crowding maior)."""
    candidates = rng_py.sample(pop, TOURNAMENT_K)
    best = candidates[0]
    for c in candidates[1:]:
        if (c.rank < best.rank
                or (c.rank == best.rank and c.crowding > best.crowding)):
            best = c
    return best


def ox_crossover(
    p1: Individual, p2: Individual, rng_py: random.Random
) -> Individual:
    """
    Order Crossover (OX) para permutações [Davis, 1985].
    Preserva uma sub-sequência de p1 e preenche o restante com a ordem de p2.
    """
    n    = len(p1.chromosome)
    a, b = sorted(rng_py.sample(range(n), 2))

    child = [-1] * n
    child[a : b + 1] = p1.chromosome[a : b + 1]
    segment = set(child[a : b + 1])

    fill = [g for g in p2.chromosome if g not in segment]
    pos  = [(b + 1 + i) % n for i in range(n - (b - a + 1))]
    for i, p in enumerate(pos):
        child[p] = fill[i]

    return Individual(chromosome=child)


def mutate(ind: Individual, rng_py: random.Random) -> Individual:
    """
    Mutação com dois operadores de igual probabilidade:
      - 2-opt intra: inverte uma sub-sequência (explora vizinhança local)
      - or-opt(1): remove e reinsere um gene em posição aleatória
    Implementação baseada em Laporte et al. (2000).
    """
    chrom = ind.chromosome[:]
    n     = len(chrom)
    if n < 3:
        return Individual(chromosome=chrom)

    if rng_py.random() < 0.5:
        # 2-opt: inversão de sub-segmento
        i, j = sorted(rng_py.sample(range(n), 2))
        chrom[i : j + 1] = chrom[i : j + 1][::-1]
    else:
        # or-opt(1): relocação de elemento
        i    = rng_py.randrange(n)
        gene = chrom.pop(i)
        j    = rng_py.randrange(n)
        chrom.insert(j, gene)

    return Individual(chromosome=chrom)


# ─────────────────────────────────────────────────────────────────────────────
# 7. INICIALIZAÇÃO DA POPULAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _nn_chromosome(
    inst: dict,
    dist: List[List[float]],
    rng_py: random.Random,
) -> List[int]:
    """
    Heurística de vizinho mais próximo com perturbação biased-randomized.
    Com probabilidade 0.15, escolhe aleatoriamente entre os 3 clientes
    mais próximos (em vez do mais próximo exato).
    Ref.: Juan et al. (2021), Seção 3.
    """
    n         = len(inst['customers'])
    unvisited = list(range(1, n + 1))
    rng_py.shuffle(unvisited)
    chrom   = []
    current = 0

    while unvisited:
        by_dist = sorted(unvisited, key=lambda c: dist[current][c])
        k       = min(3, len(by_dist))
        chosen  = rng_py.choice(by_dist[:k]) if rng_py.random() < 0.15 else by_dist[0]
        chrom.append(chosen)
        unvisited.remove(chosen)
        current = chosen

    return chrom


def _edd_chromosome(inst: dict) -> List[int]:
    """
    Cromossomo ordenado por Earliest Due Date (EDD).
    Garante que o decodificador produza poucas rotas respeitando janelas.
    Usado para semear a população inicial com soluções estruturalmente boas.
    """
    customers = inst['customers']
    return sorted(range(1, len(customers) + 1),
                  key=lambda c: customers[c - 1]['due_date_min'])


def _edd_perturbed(inst: dict, rng_py: random.Random, perturb: float = 0.15) -> List[int]:
    """
    EDD com perturbação: cada cliente tem probabilidade `perturb` de trocar
    posição com um vizinho aleatório. Cria diversidade mantendo estrutura EDD.
    """
    chrom = _edd_chromosome(inst)
    n = len(chrom)
    for i in range(n):
        if rng_py.random() < perturb:
            j = rng_py.randrange(n)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def init_population(
    inst:    dict,
    dist:    List[List[float]],
    ttime:   List[List[float]],
    pop_size: int,
    rng_py:  random.Random,
    rng_np:  np.random.Generator,
) -> List[Individual]:
    """
    Inicializa população com três componentes:
      - 20% EDD perturbado (soluções estruturalmente eficientes em janelas de tempo)
      - 40% vizinho mais próximo biased-randomized (Solomon-style)
      - 40% permutação aleatória pura (diversidade)

    Toda solução é avaliada antes de retornar.
    """
    n      = len(inst['customers'])
    n_edd  = max(1, pop_size // 5)         # 20% EDD perturbado
    n_nn   = pop_size // 2 - n_edd         # 40% vizinho mais próximo
    n_rnd  = pop_size - n_edd - n_nn       # 40% aleatório
    pop    = []

    # Soluções EDD perturbadas
    for _ in range(n_edd):
        chrom = _edd_perturbed(inst, rng_py)
        ind   = Individual(chromosome=chrom)
        evaluate(ind, inst, dist, ttime, rng_np)
        pop.append(ind)

    # Soluções vizinho mais próximo
    for _ in range(n_nn):
        chrom = _nn_chromosome(inst, dist, rng_py)
        ind   = Individual(chromosome=chrom)
        evaluate(ind, inst, dist, ttime, rng_np)
        pop.append(ind)

    # Soluções aleatórias puras
    for _ in range(n_rnd):
        chrom = list(range(1, n + 1))
        rng_py.shuffle(chrom)
        ind   = Individual(chromosome=chrom)
        evaluate(ind, inst, dist, ttime, rng_np)
        pop.append(ind)

    return pop


# ─────────────────────────────────────────────────────────────────────────────
# 8. MÉTRICAS DE QUALIDADE MULTIOBJETIVO
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(pareto: List[Individual]) -> np.ndarray:
    """Normaliza objetivos da frente para [0, 1] pelo range observado."""
    objs = np.array([ind.obj() for ind in pareto], dtype=float)
    mn   = objs.min(axis=0)
    mx   = objs.max(axis=0)
    rng  = np.where(mx > mn, mx - mn, 1.0)
    return (objs - mn) / rng


def _nondom_2d(pts: list) -> list:
    """Pontos não-dominados em 2D (minimização), ordenados por f1."""
    pts_s   = sorted(pts, key=lambda p: (p[0], p[1]))
    nd      = []
    best_f2 = float('inf')
    for p in pts_s:
        if p[1] < best_f2:
            nd.append(p)
            best_f2 = p[1]
    return nd


def _hv2d(pts: list, ref: np.ndarray) -> float:
    """Hipervolume 2D por varredura (sweep line) em relação ao ponto de referência."""
    hv      = 0.0
    prev_f1 = ref[0]
    for p in sorted(pts, key=lambda x: x[0], reverse=True):
        if p[0] < prev_f1 and p[1] < ref[1]:
            hv     += (prev_f1 - p[0]) * (ref[1] - p[1])
            prev_f1 = p[0]
    return hv


def compute_hypervolume(
    pareto:    List[Individual],
    ref_point: Optional[Tuple] = None,
) -> float:
    """
    Hipervolume 3D (indicador S de Zitzler & Thiele, 1999).
    Algoritmo: decomposição por fatias em f3 com HV 2D por varredura.
    Ponto de referência: nadir × 1.1 (ou valor fornecido pelo usuário).
    """
    if len(pareto) < 2:
        return 0.0

    objs = np.array([ind.obj() for ind in pareto], dtype=float)
    ref  = (objs.max(axis=0) * 1.1 + 1e-9
            if ref_point is None else np.array(ref_point, dtype=float))

    # Ordena por f3 decrescente (fatias)
    idx      = np.argsort(objs[:, 2])[::-1]
    sorted_o = objs[idx]

    hv       = 0.0
    prev_f3  = ref[2]
    front_2d = []

    for row in sorted_o:
        front_2d.append(row[:2])
        front_2d = _nondom_2d(front_2d)
        hv2d     = _hv2d(front_2d, ref[:2])
        delta    = prev_f3 - row[2]
        hv      += hv2d * max(delta, 0.0)
        prev_f3  = row[2]

    return float(hv)


def compute_gd(
    pareto:   List[Individual],
    ref_set:  List[Individual],
) -> float:
    """
    Generational Distance (GD) — distância média da frente calculada
    ao conjunto de referência, usando objetivos normalizados.
    GD = (1/|P|) Σᵢ min_{r ∈ R} ‖pᵢ − r‖₂
    """
    if not pareto or not ref_set:
        return float('inf')
    p_objs = _normalize(pareto)
    r_objs = _normalize(ref_set)
    dists  = [min(float(np.linalg.norm(p - r)) for r in r_objs) for p in p_objs]
    return float(np.mean(dists))


def compute_spread(pareto: List[Individual]) -> float:
    """
    Spread (Δ) — diversidade da frente de Pareto.
    Métrica de Deb et al. (2002), estendida para 3 objetivos:
    as soluções são ordenadas por f1 (custo) e as distâncias entre
    vizinhos consecutivos são comparadas com sua média.

    Δ = (d_f + d_l + Σ|dᵢ − d̄|) / (d_f + d_l + (n−1)d̄)
    """
    if len(pareto) < 3:
        return 0.0

    objs_s = _normalize(pareto)
    objs_s = objs_s[np.argsort(objs_s[:, 0])]
    dists  = [float(np.linalg.norm(objs_s[i + 1] - objs_s[i]))
               for i in range(len(objs_s) - 1)]
    if not dists:
        return 0.0

    d_bar = float(np.mean(dists))
    d_f, d_l = dists[0], dists[-1]
    num  = d_f + d_l + sum(abs(d - d_bar) for d in dists)
    den  = d_f + d_l + len(dists) * d_bar
    return float(num / den) if den > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. LOOP PRINCIPAL DO NSGA-II
# ─────────────────────────────────────────────────────────────────────────────

def run_nsga2(
    inst:       dict,
    pop_size:   int  = POP_SIZE,
    n_gen:      int  = N_GEN,
    mc_samples: int  = MC_SAMPLES,
    seed:       int  = 0,
    verbose:    bool = True,
) -> Tuple[List[Individual], List[Tuple[int, float]]]:
    """
    Executa o NSGA-II com Monte Carlo embarcado para o CCVRPTW-ST.

    A cada 10 gerações calcula o Hipervolume da frente corrente.

    Retorna:
      pareto     — frente de Pareto final
      hv_history — lista de (geração, HV) a cada 10 gerações
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    dist, ttime = build_matrices(inst)

    if verbose:
        print(f"  Inicializando população ({pop_size} indivíduos)...", flush=True)
    pop = init_population(inst, dist, ttime, pop_size, rng_py, rng_np)

    hv_history: List[Tuple[int, float]] = []
    t0 = time.time()

    for gen in range(1, n_gen + 1):
        # Classifica a população atual
        fronts = fast_nondominated_sort(pop)
        for front in fronts:
            crowding_distance_assign(front)

        # Geração de filhos
        offspring: List[Individual] = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, rng_py)
            p2 = tournament_select(pop, rng_py)
            child = ox_crossover(p1, p2, rng_py) if rng_py.random() < CROSSOVER_RATE \
                    else Individual(chromosome=p1.chromosome[:])
            if rng_py.random() < MUTATION_RATE:
                child = mutate(child, rng_py)
            evaluate(child, inst, dist, ttime, rng_np, mc_samples)
            offspring.append(child)

        # União (pais + filhos) e seleção elitista — Deb et al. (2002), Seção III-C
        combined = pop + offspring
        fronts   = fast_nondominated_sort(combined)
        for front in fronts:
            crowding_distance_assign(front)

        new_pop: List[Individual] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                remaining = pop_size - len(new_pop)
                sf = sorted(front, key=lambda x: -x.crowding)
                new_pop.extend(sf[:remaining])
                break
        pop = new_pop

        # Hipervolume a cada 10 gerações
        if gen % 10 == 0 or gen == n_gen:
            pf = [ind for ind in pop if ind.rank == 1]
            hv = compute_hypervolume(pf)
            hv_history.append((gen, hv))
            if verbose:
                elapsed = time.time() - t0
                n_feas  = sum(1 for ind in pf if ind.feasible)
                print(
                    f"  Gen {gen:4d}/{n_gen} | Pareto: {len(pf):3d} sol "
                    f"({n_feas} viáveis) | HV: {hv:.4e} | {elapsed:.1f}s",
                    flush=True,
                )

    pareto = [ind for ind in pop if ind.rank == 1]
    return pareto, hv_history


# ─────────────────────────────────────────────────────────────────────────────
# 10. SAÍDA DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    pareto:     List[Individual],
    hv_history: List[Tuple[int, float]],
    metrics:    Dict,
    inst_name:  str,
    output_dir: str,
):
    """Salva frente de Pareto (CSV), histórico de HV (CSV) e métricas (JSON)."""
    os.makedirs(output_dir, exist_ok=True)

    # Frente de Pareto
    pf_path = os.path.join(output_dir, f"{inst_name}_pareto.csv")
    with open(pf_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['sol_id', 'f1_cost_BRL', 'f2_quality_loss',
                    'f3_co2_kg', 'n_routes', 'feasible', 'routes'])
        for i, ind in enumerate(sorted(pareto, key=lambda x: x.f_cost)):
            w.writerow([i + 1, round(ind.f_cost, 2), round(ind.f_qloss, 6),
                        round(ind.f_co2, 4), len(ind.routes), ind.feasible,
                        str(ind.routes)])

    # Histórico de hipervolume
    hv_path = os.path.join(output_dir, f"{inst_name}_hv_history.csv")
    with open(hv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['generation', 'hypervolume'])
        for gen, hv in hv_history:
            w.writerow([gen, round(hv, 8)])

    # Métricas finais
    m_path = os.path.join(output_dir, f"{inst_name}_metrics.json")
    with open(m_path, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    print(f"  Salvo: {pf_path}")
    print(f"         {hv_path}")
    print(f"         {m_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. RUNNER POR INSTÂNCIA
# ─────────────────────────────────────────────────────────────────────────────

def run_instance(
    instance_path: str,
    output_dir:    str,
    seed:          int  = 0,
    pop_size:      int  = POP_SIZE,
    n_gen:         int  = N_GEN,
    mc_samples:    int  = MC_SAMPLES,
    verbose:       bool = True,
) -> Tuple[List[Individual], Dict]:
    """Carrega instância, executa NSGA-II e salva resultados."""
    inst      = load_instance(instance_path)
    inst_name = inst['name']

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Instância : {inst_name}")
        print(f"  Família   : {inst['family']}   n={inst['size']}   "
              f"Incerteza: {inst['uncertainty_level'].upper()}")
        print(f"  σ_T={inst['thermal_profile']['sigma_T_C']}°C  |  "
              f"{inst['fleet']['num_vehicles']} veículos × "
              f"{inst['fleet']['capacity_kg']:.0f} kg")
        print(f"{'='*65}")

    t_start = time.time()
    pareto, hv_history = run_nsga2(
        inst, pop_size=pop_size, n_gen=n_gen,
        mc_samples=mc_samples, seed=seed, verbose=verbose,
    )
    elapsed = time.time() - t_start

    hv_final = compute_hypervolume(pareto)
    spread   = compute_spread(pareto)
    gd       = compute_gd(pareto, pareto)    # auto-referência (sem Pareto verdadeiro)

    metrics = {
        "instance":          inst_name,
        "n_customers":       inst['size'],
        "family":            inst['family'],
        "uncertainty":       inst['uncertainty_level'],
        "sigma_T_C":         inst['thermal_profile']['sigma_T_C'],
        "algorithm":         "NSGA-II + Monte Carlo (Simheurística)",
        "pop_size":          pop_size,
        "n_generations":     n_gen,
        "mc_samples":        mc_samples,
        "seed":              seed,
        "elapsed_seconds":   round(elapsed, 2),
        "pareto_front_size": len(pareto),
        "n_feasible":        sum(1 for ind in pareto if ind.feasible),
        "hypervolume":       round(hv_final, 8),
        "spread_delta":      round(spread, 6),
        "gd_self":           round(gd, 6),
        "best_cost_BRL":     round(min(ind.f_cost  for ind in pareto), 2),
        "best_quality_pct":  round((1 - min(ind.f_qloss for ind in pareto)) * 100, 3),
        "best_co2_kg":       round(min(ind.f_co2   for ind in pareto), 4),
    }

    save_results(pareto, hv_history, metrics, inst_name, output_dir)

    if verbose:
        print(f"\n  ✓ Concluído em {elapsed:.1f}s")
        print(f"  Frente Pareto : {len(pareto)} soluções "
              f"({metrics['n_feasible']} viáveis)")
        print(f"  Hipervolume   : {hv_final:.4e}")
        print(f"  Spread (Δ)    : {spread:.4f}")
        print(f"  Melhor custo  : R${metrics['best_cost_BRL']:,.2f}")
        print(f"  Melhor qualid.: {metrics['best_quality_pct']:.2f}%")
        print(f"  Melhor CO₂    : {metrics['best_co2_kg']:.2f} kg CO₂-eq")

    return pareto, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 11b. BUSCA ALEATÓRIA MULTIOBJETIVO (BASELINE INFERIOR)
# ─────────────────────────────────────────────────────────────────────────────

def _update_pareto_archive(archive: List[Individual], candidate: Individual) -> List[Individual]:
    """Mantém arquivo de soluções não-dominadas."""
    new_archive = [ind for ind in archive if not candidate.dominates(ind)]
    if all(not ind.dominates(candidate) for ind in archive):
        new_archive.append(candidate)
    return new_archive


def run_random_search(
    inst:       dict,
    pop_size:   int  = POP_SIZE,
    n_gen:      int  = N_GEN,
    mc_samples: int  = MC_SAMPLES,
    seed:       int  = 0,
    verbose:    bool = True,
) -> Tuple[List[Individual], List[Tuple[int, float]]]:
    """
    Busca aleatória multiobjetivo com Monte Carlo (MOEA-RS).
    Gera pop_size permutações aleatórias por iteração e mantém arquivo Pareto.
    Serve como baseline inferior: sem inteligência evolucionária.
    """
    rng_py = random.Random(seed)
    rng_np = np.random.default_rng(seed)
    dist, ttime = build_matrices(inst)
    n = len(inst['customers'])

    archive: List[Individual] = []
    hv_history: List[Tuple[int, float]] = []
    t0 = time.time()

    for gen in range(1, n_gen + 1):
        for _ in range(pop_size):
            chrom = list(range(1, n + 1))
            rng_py.shuffle(chrom)
            ind = Individual(chromosome=chrom)
            evaluate(ind, inst, dist, ttime, rng_np, mc_samples)
            archive = _update_pareto_archive(archive, ind)

        if gen % 10 == 0 or gen == n_gen:
            hv = compute_hypervolume(archive)
            hv_history.append((gen, hv))
            if verbose:
                print(f"  Iter {gen:4d}/{n_gen} | Arquivo: {len(archive):3d} sol | HV: {hv:.4e} | {time.time()-t0:.1f}s", flush=True)

    return archive, hv_history


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    INST_DIR    = os.path.normpath(os.path.join(BASE_DIR, '..', 'instances', 'generated'))
    RESULTS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'results'))

    # Argumento opcional: caminho da instância
    if len(sys.argv) > 1:
        inst_path = sys.argv[1]
    else:
        # Por padrão: menor instância (n=25, família C1, incerteza baixa)
        inst_path = os.path.join(INST_DIR, 'CCVRPTW-ST_C1_025_LOW.json')

    if not os.path.exists(inst_path):
        print(f"Arquivo não encontrado: {inst_path}")
        sys.exit(1)

    run_instance(inst_path, RESULTS_DIR, seed=42, verbose=True)
