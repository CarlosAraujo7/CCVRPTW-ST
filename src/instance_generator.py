"""
CCVRPTW-ST Instance Generator
==============================
Cold Chain VRP with Time Windows and Stochastic Temperature

Gera instancias benchmark derivadas da metodologia de Solomon (1987) com
parametros termicos fundamentados na literatura de ciencia de alimentos.

Metodologia de geracao baseada em:
  - Solomon, M.M. (1987). Algorithms for the Vehicle Routing and Scheduling
    Problems with Time Window Constraints. Operations Research, 35(2), 254-265.
  - Arrhenius, S. (1889). Uber die Reaktionsgeschwindigkeit bei der Inversion
    von Rohrzucker durch Sauren. Z. Phys. Chem., 4, 226-248.
  - Gwanpua, S.G. et al. (2015). The FRISBEE tool, a software for optimising
    the trade-off between food quality, energy use, and global warming impact
    of cold chains. Journal of Food Engineering, 148, 2-12.
  - Ndraha, N. et al. (2018). Time-temperature abuse in the food cold chain:
    Review of issues, challenges, and recommendations. Food Control, 89, 12-21.
  - Zhang, J. et al. (2023). Optimal distribution of perishable foods with
    storage temperature control and quality requirements. Computers &
    Industrial Engineering, 178, 109146.

Autor: Gerado automaticamente — Maio/2026
Uso  : python instance_generator.py
"""

import json
import math
import random
import csv
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARAMETROS TERMICOS — fundamentados na literatura de ciencia de alimentos
# ─────────────────────────────────────────────────────────────────────────────
# Cinetica de Arrhenius:  k(T) = A0 * exp(-Ea / (R * T))
#   onde T esta em Kelvin, R = 8.314 J/(mol*K)
#
# Referencias para cada categoria:
#   Carne bovina:  Ndraha et al. (2018); Giannakourou & Taoukis (2003)
#   Laticinios:    Gwanpua et al. (2015); Taoukis et al. (1997)
#   Hortifruti:    Jedermann et al. (2014); Hertog et al. (2007)
#   Congelados:    Van der Sman et al. (2012); ASHRAE Fundamentals (2017)
# ─────────────────────────────────────────────────────────────────────────────

R_GAS = 8.314  # J / (mol * K)

THERMAL_PROFILES = {
    "meat": {
        "description": "Carne bovina resfriada",
        "T_set_C": 2.0,          # temperatura nominal de transporte (°C)
        "T_max_C": 7.0,          # temperatura maxima sanitaria (°C)
        "T_min_C": -1.5,         # temperatura minima (abaixo: congelamento parcial)
        "Ea_J_mol": 75_000.0,    # energia de ativacao (J/mol) — Ndraha et al. 2018
        # A0 recalibrado para k(T_set) => vida util = 240h (10 dias) com Q_end=0.01
        # Metodologia: A0 = [-ln(0.01)/shelf_h] / exp(-Ea/(R*T_K))
        "A0": 3.3235e12,
        "Q_min": 0.70,           # qualidade residual minima aceitavel para entrega
        "Q_end_of_life": 0.01,   # limiar de fim de vida util (referencia de calibracao)
        "unit_value_BRL_kg": 35.0,
        "shelf_life_h_at_Tset": 240,  # vida util a T_set (10 dias) — referencia
        "references": [
            "Ndraha et al. (2018) Food Control 89:12-21",
            "Giannakourou & Taoukis (2003) J. Food Sci. 68(1):201-209"
        ]
    },
    "dairy": {
        "description": "Laticinios (iogurte, queijo fresco, leite UHT refrigerado)",
        "T_set_C": 4.0,
        "T_max_C": 8.0,
        "T_min_C": 0.0,
        "Ea_J_mol": 62_000.0,    # Gwanpua et al. (2015) J. Food Eng. 148:2-12
        # A0 recalibrado: vida util = 168h (7 dias), Q_end=0.01
        "A0": 1.3290e10,
        "Q_min": 0.75,
        "Q_end_of_life": 0.01,
        "unit_value_BRL_kg": 18.0,
        "shelf_life_h_at_Tset": 168,  # 7 dias
        "references": [
            "Gwanpua et al. (2015) J. Food Engineering 148:2-12",
            "Taoukis et al. (1997) Crit. Rev. Food Sci. Nutr. 37(4):359-394"
        ]
    },
    "produce": {
        "description": "Hortifruti (folhosos, frutas tropicais, verduras)",
        "T_set_C": 5.0,
        "T_max_C": 10.0,
        "T_min_C": 1.0,
        "Ea_J_mol": 55_000.0,    # Jedermann et al. (2014); Hertog et al. (2007)
        # A0 recalibrado: vida util = 120h (5 dias), Q_end=0.01
        "A0": 8.1857e8,
        "Q_min": 0.65,
        "Q_end_of_life": 0.01,
        "unit_value_BRL_kg": 8.0,
        "shelf_life_h_at_Tset": 120,  # 5 dias
        "references": [
            "Jedermann et al. (2014) Philos. Trans. R. Soc. A 372:20130302",
            "Hertog et al. (2007) Postharvest Biol. Technol. 45:2-11"
        ]
    },
    "frozen": {
        "description": "Congelados (frango, peixe, sorvetes)",
        "T_set_C": -18.0,
        "T_max_C": -15.0,
        "T_min_C": -25.0,
        "Ea_J_mol": 105_000.0,   # Van der Sman et al. (2012); ASHRAE (2017)
        # A0 recalibrado: vida util = 2160h (90 dias), Q_end=0.01
        "A0": 6.6881e18,
        "Q_min": 0.80,
        "Q_end_of_life": 0.01,
        "unit_value_BRL_kg": 22.0,
        "shelf_life_h_at_Tset": 2160,  # 90 dias
        "references": [
            "Van der Sman et al. (2012) Food Biophysics 7(1):1-10",
            "ASHRAE Fundamentals Handbook (2017) Chapter 22"
        ]
    }
}

# Niveis de incerteza termica (desvio-padrao das flutuacoes de temperatura)
UNCERTAINTY_LEVELS = {
    "low":    {"sigma_T": 0.5,  "sigma_t_pct": 0.05, "label": "Baixa incerteza"},
    "medium": {"sigma_T": 1.0,  "sigma_t_pct": 0.10, "label": "Media incerteza"},
    "high":   {"sigma_T": 2.0,  "sigma_t_pct": 0.20, "label": "Alta incerteza"},
}

# Parametros de emissao de CO2
# Fonte: IPCC (2021); Ecoinvent v3 (2022); valores tipicos para veiculos refrigerados leves
CO2_PARAMS = {
    "alpha_kg_per_km": 0.210,      # emissao do motor por km (kg CO2/km)
    "beta_kg_per_kWh": 0.092,      # emissao do sistema de refrigeracao por kWh (kg CO2/kWh)
    "P_refrigeration_kW": 3.5,     # potencia tipica do sistema de refrigeracao
    "references": [
        "IPCC (2021) AR6 WGI Chapter 7",
        "Ecoinvent v3.8 (2022) — transporte refrigerado"
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float           # kg
    ready_time: float       # minutos desde o inicio do horizonte
    due_date: float         # minutos
    service_time: float     # minutos
    product_category: str   # "meat", "dairy", "produce", "frozen"
    unit_value_BRL_kg: float


@dataclass
class Depot:
    id: int = 0
    x: float = 0.0
    y: float = 0.0
    ready_time: float = 0.0
    due_date: float = 960.0  # 16 horas de horizonte


@dataclass
class Fleet:
    num_vehicles: int
    capacity_kg: float
    speed_km_h: float = 60.0
    fixed_cost_BRL: float = 150.0   # custo fixo por veiculo utilizado
    variable_cost_BRL_km: float = 2.50  # custo variavel por km


@dataclass
class ThermalProfile:
    category: str
    T_set_C: float
    T_max_C: float
    T_min_C: float
    Ea_J_mol: float
    A0: float
    Q_min: float
    sigma_T: float        # incerteza termica (desvio-padrao, °C)
    sigma_t_pct: float    # incerteza de tempo de percurso (% do tempo nominal)


@dataclass
class Instance:
    name: str
    family: str           # C, R, RC
    size: int             # 25, 50, 100
    uncertainty_level: str  # low, medium, high
    depot: Dict
    customers: List[Dict]
    fleet: Dict
    thermal_profile: Dict
    co2_params: Dict
    horizon_minutes: float
    generation_metadata: Dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. GERACAO DE COORDENADAS — metodologia Solomon (1987)
# ─────────────────────────────────────────────────────────────────────────────

def euclidean(x1, y1, x2, y2) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def travel_time_min(x1, y1, x2, y2, speed_km_h=60.0) -> float:
    """Converte distancia euclidiana (em unidades Solomon = km) para minutos."""
    dist = euclidean(x1, y1, x2, y2)
    return (dist / speed_km_h) * 60.0


def generate_C_coordinates(n: int, rng: random.Random) -> List[Tuple[float, float]]:
    """
    Familia C (Clustered): clientes agrupados em clusters.
    Solomon usou ~10 clusters para n=100. Escalamos proporcionalmente.
    Grade: [0, 100] x [0, 100], deposito em (40, 50).
    """
    n_clusters = max(3, n // 10)
    centers = [(rng.uniform(10, 90), rng.uniform(10, 90)) for _ in range(n_clusters)]
    coords = []
    per_cluster = n // n_clusters
    remainder = n % n_clusters
    for ci, (cx, cy) in enumerate(centers):
        count = per_cluster + (1 if ci < remainder else 0)
        for _ in range(count):
            x = min(100, max(0, rng.gauss(cx, 8)))
            y = min(100, max(0, rng.gauss(cy, 8)))
            coords.append((round(x, 2), round(y, 2)))
    rng.shuffle(coords)
    return coords[:n]


def generate_R_coordinates(n: int, rng: random.Random) -> List[Tuple[float, float]]:
    """
    Familia R (Random): clientes uniformemente distribuidos em [0, 100]^2.
    """
    return [(round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2))
            for _ in range(n)]


def generate_RC_coordinates(n: int, rng: random.Random) -> List[Tuple[float, float]]:
    """
    Familia RC (Random-Clustered): metade agrupada, metade aleatoria.
    """
    n_cluster = n // 2
    n_random = n - n_cluster
    clustered = generate_C_coordinates(n_cluster, rng)
    random_pts = generate_R_coordinates(n_random, rng)
    coords = clustered + random_pts
    rng.shuffle(coords)
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# 4. GERACAO DE JANELAS DE TEMPO — Solomon (1987)
# ─────────────────────────────────────────────────────────────────────────────

def generate_time_windows(
    coords: List[Tuple], depot_x: float, depot_y: float,
    horizon: float, tw_width: float, rng: random.Random, speed: float = 60.0
) -> List[Tuple[float, float, float]]:
    """
    Gera janelas de tempo no estilo Solomon:
      - earliest = tempo de deslocamento do deposito ate o cliente (com folga)
      - latest = earliest + tw_width
      - service_time: 10-20 minutos

    tw_width controla a "largura" da janela:
      - Tipo 1 (C1, R1, RC1): janelas estreitas (~45-90 min)
      - Tipo 2 (C2, R2, RC2): janelas largas (~240-480 min)

    Retorna lista de (ready_time, due_date, service_time).
    """
    windows = []
    for (x, y) in coords:
        dist = euclidean(depot_x, depot_y, x, y)
        t_travel = (dist / speed) * 60.0
        # Folga aleatoria para o ready_time
        slack = rng.uniform(0, max(0, horizon * 0.3 - t_travel))
        ready = round(t_travel + slack, 1)
        due = round(min(horizon, ready + rng.uniform(tw_width * 0.7, tw_width * 1.3)), 1)
        service = round(rng.uniform(10, 20), 1)
        windows.append((ready, due, service))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# 5. GERACAO DE DEMANDAS E CATEGORIAS DE PRODUTO
# ─────────────────────────────────────────────────────────────────────────────

DEMAND_RANGES = {
    "meat":    (50, 200),   # kg por cliente
    "dairy":   (30, 150),
    "produce": (20, 100),
    "frozen":  (40, 180),
}

def assign_product_categories(n: int, rng: random.Random) -> List[str]:
    """
    Distribui categorias de produto entre os clientes.
    Distribuicao tipica de um distribuidor de alimentos brasileiro:
      meat: 35%, dairy: 25%, produce: 25%, frozen: 15%
    Fonte: ABIA (Associacao Brasileira das Industrias da Alimentacao, 2023)
    """
    cats = (
        ["meat"] * round(n * 0.35) +
        ["dairy"] * round(n * 0.25) +
        ["produce"] * round(n * 0.25) +
        ["frozen"] * round(n * 0.15)
    )
    # Ajusta para n exato
    while len(cats) < n:
        cats.append(rng.choice(["meat", "dairy", "produce"]))
    cats = cats[:n]
    rng.shuffle(cats)
    return cats


def generate_demands(categories: List[str], rng: random.Random) -> List[float]:
    return [round(rng.uniform(*DEMAND_RANGES[c]), 1) for c in categories]


# ─────────────────────────────────────────────────────────────────────────────
# 6. CALCULO DA TAXA DE DEGRADACAO DE ARRHENIUS
# ─────────────────────────────────────────────────────────────────────────────

def arrhenius_rate(T_celsius: float, Ea_J_mol: float, A0: float) -> float:
    """
    k(T) = A0 * exp(-Ea / (R * T_K))
    Retorna a taxa de degradacao em 1/hora.
    """
    T_K = T_celsius + 273.15
    return A0 * math.exp(-Ea_J_mol / (R_GAS * T_K))


def quality_after_time(T_celsius: float, duration_h: float,
                        Ea_J_mol: float, A0: float) -> float:
    """
    Qualidade residual apos 'duration_h' horas a temperatura T_celsius.
    Modelo de primeira ordem: Q(t) = exp(-k(T) * t)
    """
    k = arrhenius_rate(T_celsius, Ea_J_mol, A0)
    return math.exp(-k * duration_h)


def validate_thermal_params(profile: dict) -> dict:
    """
    Valida e documenta os parametros termicos calculados.
    Usa Q_end_of_life = 0.01 como criterio de fim de vida util:
      shelf_life = -ln(Q_end) / k(T_set)
    """
    cat = profile["category"]
    tp = THERMAL_PROFILES[cat]
    Q_end = tp.get("Q_end_of_life", 0.01)
    k_at_Tset = arrhenius_rate(tp["T_set_C"], tp["Ea_J_mol"], tp["A0"])
    k_at_Tmax = arrhenius_rate(tp["T_max_C"], tp["Ea_J_mol"], tp["A0"])
    # Vida util: Q(t) = exp(-k*t) => t = -ln(Q_end)/k
    shelf_life_calc = -math.log(Q_end) / k_at_Tset if k_at_Tset > 0 else float("inf")
    shelf_life_at_Tmax = -math.log(Q_end) / k_at_Tmax if k_at_Tmax > 0 else float("inf")
    # Tempo ate atingir Q_min durante a rota (max tolerado na entrega)
    t_to_Qmin = -math.log(tp["Q_min"]) / k_at_Tset if k_at_Tset > 0 else float("inf")
    # Qualidade apos rota tipica de 8h e 16h a T_set
    Q_after_8h  = math.exp(-k_at_Tset * 8.0)
    Q_after_16h = math.exp(-k_at_Tset * 16.0)
    return {
        "k_at_Tset_1_per_h": round(k_at_Tset, 6),
        "k_at_Tmax_1_per_h": round(k_at_Tmax, 6),
        "ratio_kmax_over_kset": round(k_at_Tmax / k_at_Tset, 2),
        "shelf_life_at_Tset_h_calculated": round(shelf_life_calc, 1),
        "shelf_life_at_Tmax_h_calculated": round(shelf_life_at_Tmax, 1),
        "shelf_life_at_Tset_h_literature": tp["shelf_life_h_at_Tset"],
        "max_route_h_to_reach_Qmin": round(t_to_Qmin, 1),
        "Q_after_8h_at_Tset": round(Q_after_8h, 4),
        "Q_after_16h_at_Tset": round(Q_after_16h, 4),
        "calibration_note": (
            f"A0 calibrado para k(T_set)*shelf_h = -ln({Q_end}); "
            f"metodologia: A0 = [-ln({Q_end})/shelf_h] / exp(-Ea/(R*T_K))"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. ESTIMATIVA DE FROTA POR JANELA DE TEMPO
# ─────────────────────────────────────────────────────────────────────────────

def estimate_routes_by_tw(
    coords: List[Tuple[float, float]],
    time_windows: List[Tuple[float, float, float]],   # (ready, due, service) em min
    demands: List[float],
    capacity: float,
    speed_km_h: float,
    depot_x: float,
    depot_y: float,
) -> int:
    """
    Estima o numero minimo de rotas necessario para atender todos os clientes
    respeitando janelas de tempo e capacidade, usando heuristica greedy EDD
    (Earliest Due Date first).

    Usada para garantir que o numero de veiculos e suficiente tambem do ponto
    de vista das janelas de tempo (nao apenas da capacidade de carga).

    Complexidade O(n^2) — aceitavel para n <= 100.
    """
    nodes = [(depot_x, depot_y)] + list(coords)

    def tt(i: int, j: int) -> float:
        """Tempo de percurso em minutos entre nos i e j."""
        dx = nodes[i][0] - nodes[j][0]
        dy = nodes[i][1] - nodes[j][1]
        return math.hypot(dx, dy) / speed_km_h * 60.0

    n = len(coords)
    # Ordena por due_date (EDD)
    order = sorted(range(n), key=lambda i: time_windows[i][1])

    r_load: List[float] = []
    r_time: List[float] = []
    r_last: List[int]   = []

    for idx in order:
        cid = idx + 1           # indice no sistema de nos (1-based)
        dem = demands[idx]
        rt, due, st = time_windows[idx]
        inserted = False

        for k in range(len(r_load)):
            if r_load[k] + dem > capacity:
                continue
            travel  = tt(r_last[k], cid)
            arrival = r_time[k] + travel
            if arrival > due:
                continue
            wait       = max(0.0, rt - arrival)
            r_load[k] += dem
            r_time[k]  = arrival + wait + st
            r_last[k]  = cid
            inserted   = True
            break

        if not inserted:
            travel  = tt(0, cid)
            arrival = travel
            wait    = max(0.0, rt - arrival)
            r_load.append(dem)
            r_time.append(arrival + wait + st)
            r_last.append(cid)

    return max(1, len(r_load))


# ─────────────────────────────────────────────────────────────────────────────
# 8. GERADOR PRINCIPAL DE INSTANCIAS
# ─────────────────────────────────────────────────────────────────────────────

def generate_instance(
    family: str,        # "C", "R", "RC"
    size: int,          # 25, 50, 100
    uncertainty: str,   # "low", "medium", "high"
    seed: int,
    variant: int = 1,   # variante dentro da familia (1 ou 2: janela estreita/larga)
) -> Instance:

    rng = random.Random(seed)
    depot_x, depot_y = 40.0, 50.0
    horizon = 960.0  # 16 horas em minutos

    # Largura de janela de tempo por variante
    tw_width = 75.0 if variant == 1 else 360.0

    # Coordenadas por familia
    if family == "C":
        coords = generate_C_coordinates(size, rng)
    elif family == "R":
        coords = generate_R_coordinates(size, rng)
    else:  # RC
        coords = generate_RC_coordinates(size, rng)

    # Janelas de tempo
    time_windows = generate_time_windows(
        coords, depot_x, depot_y, horizon, tw_width, rng
    )

    # Categorias e demandas
    categories = assign_product_categories(size, rng)
    demands = generate_demands(categories, rng)

    # Dimensionamento da frota:
    # Capacidade por veiculo = caminhao refrigerado leve tipico (600-800 kg)
    # Fonte: ANTT (2023) tabela de frete; tipico mercado brasileiro de distribuicao
    #
    # num_veiculos = max(criterio_capacidade, criterio_janela_tempo)
    # garantindo factibilidade tanto pela carga quanto pelas restricoes temporais.
    total_demand = sum(demands)
    capacity     = round(rng.uniform(600, 800), 0)   # kg/veiculo
    target_occ   = rng.uniform(0.68, 0.88)            # taxa de ocupacao alvo

    n_cap = max(2, math.ceil(total_demand / (capacity * target_occ)))
    n_tw  = estimate_routes_by_tw(
        coords, time_windows, demands, capacity, 60.0, depot_x, depot_y
    )
    # Buffer de 10% sobre o criterio de janela (arredondado para cima)
    # para permitir margem de manobra ao algoritmo de otimizacao
    num_vehicles = max(n_cap, math.ceil(n_tw * 1.10))

    # Clientes
    customers = []
    for i, ((x, y), (rt, dd, st), cat, dem) in enumerate(
        zip(coords, time_windows, categories, demands), start=1
    ):
        tp = THERMAL_PROFILES[cat]
        customers.append({
            "id": i,
            "x": x,
            "y": y,
            "demand_kg": dem,
            "ready_time_min": rt,
            "due_date_min": dd,
            "service_time_min": st,
            "product_category": cat,
            "product_description": tp["description"],
            "unit_value_BRL_kg": tp["unit_value_BRL_kg"],
            "load_value_BRL": round(dem * tp["unit_value_BRL_kg"], 2),
        })

    # Perfil termico da instancia (usa o produto majoritario)
    from collections import Counter
    dominant_cat = Counter(categories).most_common(1)[0][0]
    tp = THERMAL_PROFILES[dominant_cat]
    unc = UNCERTAINTY_LEVELS[uncertainty]

    thermal_profile = {
        "dominant_category": dominant_cat,
        "description": tp["description"],
        "T_set_C": tp["T_set_C"],
        "T_max_C": tp["T_max_C"],
        "T_min_C": tp["T_min_C"],
        "Ea_J_mol": tp["Ea_J_mol"],
        "A0": tp["A0"],
        "Q_min": tp["Q_min"],
        "sigma_T_C": unc["sigma_T"],
        "sigma_t_pct": unc["sigma_t_pct"],
        "uncertainty_label": unc["label"],
        "validation": validate_thermal_params({"category": dominant_cat}),
        "references": tp["references"],
        # Parametros por categoria de produto
        "per_category": {
            cat: {
                "T_set_C": THERMAL_PROFILES[cat]["T_set_C"],
                "T_max_C": THERMAL_PROFILES[cat]["T_max_C"],
                "Ea_J_mol": THERMAL_PROFILES[cat]["Ea_J_mol"],
                "A0": THERMAL_PROFILES[cat]["A0"],
                "Q_min": THERMAL_PROFILES[cat]["Q_min"],
                "sigma_T_C": unc["sigma_T"],
                "validation": validate_thermal_params({"category": cat}),
            }
            for cat in set(categories)
        }
    }

    name = f"CCVRPTW-ST_{family}{variant}_{size:03d}_{uncertainty.upper()}"

    return Instance(
        name=name,
        family=f"{family}{variant}",
        size=size,
        uncertainty_level=uncertainty,
        depot={
            "id": 0, "x": depot_x, "y": depot_y,
            "ready_time_min": 0.0, "due_date_min": horizon
        },
        customers=customers,
        fleet={
            "num_vehicles": num_vehicles,
            "capacity_kg": capacity,
            "speed_km_h": 60.0,
            "fixed_cost_BRL": 150.0,
            "variable_cost_BRL_km": 2.50,
        },
        thermal_profile=thermal_profile,
        co2_params=CO2_PARAMS,
        horizon_minutes=horizon,
        generation_metadata={
            "seed": seed,
            "family": family,
            "variant": variant,
            "tw_width_min": tw_width,
            "total_demand_kg": round(total_demand, 1),
            "num_customers": size,
            "depot_position": [depot_x, depot_y],
            "category_distribution": dict(Counter(categories)),
            "solomon_reference": "Solomon (1987) Operations Research 35(2):254-265",
            "arrhenius_references": [
                "Ndraha et al. (2018) Food Control 89:12-21",
                "Gwanpua et al. (2015) J. Food Engineering 148:2-12",
                "Jedermann et al. (2014) Philos. Trans. R. Soc. A 372:20130302",
                "ASHRAE Fundamentals Handbook (2017) Chapter 22"
            ],
            "co2_references": [
                "IPCC (2021) AR6 WGI",
                "Ecoinvent v3.8 (2022)"
            ],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOTE COMPLETO — 27 INSTANCIAS
# ─────────────────────────────────────────────────────────────────────────────

INSTANCE_PLAN = [
    # ───────────────────────────────────────────────────────────────────────
    # Formato: (familia, variante, tamanho, incerteza, seed)
    #
    # Variante 1 = janelas de tempo ESTREITAS (~75 min) — cenario operacional
    #              tipico de distribuicao urbana com restricoes rigidas de horario
    # Variante 2 = janelas de tempo LARGAS (~360 min)  — cenario de distribuicao
    #              regional ou com maior flexibilidade de entrega
    #
    # Referencia estrutural: Solomon (1987) — conjuntos C1/C2, R1/R2, RC1/RC2
    # ───────────────────────────────────────────────────────────────────────

    # ── Familia C (Clustered) — variante 1: janelas estreitas ───────────────
    ("C",  1,  25, "low",    42),
    ("C",  1,  25, "medium", 43),
    ("C",  1,  25, "high",   44),
    ("C",  1,  50, "low",    52),
    ("C",  1,  50, "medium", 53),
    ("C",  1,  50, "high",   54),
    ("C",  1, 100, "low",    62),
    ("C",  1, 100, "medium", 63),
    ("C",  1, 100, "high",   64),

    # ── Familia C (Clustered) — variante 2: janelas largas ──────────────────
    ("C",  2,  25, "low",   142),
    ("C",  2,  25, "medium",143),
    ("C",  2,  25, "high",  144),
    ("C",  2,  50, "low",   152),
    ("C",  2,  50, "medium",153),
    ("C",  2,  50, "high",  154),
    ("C",  2, 100, "low",   162),
    ("C",  2, 100, "medium",163),
    ("C",  2, 100, "high",  164),

    # ── Familia R (Random) — variante 1: janelas estreitas ──────────────────
    ("R",  1,  25, "low",   242),
    ("R",  1,  25, "medium",243),
    ("R",  1,  25, "high",  244),
    ("R",  1,  50, "low",   252),
    ("R",  1,  50, "medium",253),
    ("R",  1,  50, "high",  254),
    ("R",  1, 100, "low",   262),
    ("R",  1, 100, "medium",263),
    ("R",  1, 100, "high",  264),

    # ── Familia R (Random) — variante 2: janelas largas ─────────────────────
    ("R",  2,  25, "low",   342),
    ("R",  2,  25, "medium",343),
    ("R",  2,  25, "high",  344),
    ("R",  2,  50, "low",   352),
    ("R",  2,  50, "medium",353),
    ("R",  2,  50, "high",  354),
    ("R",  2, 100, "low",   362),
    ("R",  2, 100, "medium",363),
    ("R",  2, 100, "high",  364),

    # ── Familia RC (Random-Clustered) — variante 1: janelas estreitas ────────
    ("RC", 1,  25, "low",   442),
    ("RC", 1,  25, "medium",443),
    ("RC", 1,  25, "high",  444),
    ("RC", 1,  50, "low",   452),
    ("RC", 1,  50, "medium",453),
    ("RC", 1,  50, "high",  454),
    ("RC", 1, 100, "low",   462),
    ("RC", 1, 100, "medium",463),
    ("RC", 1, 100, "high",  464),

    # ── Familia RC (Random-Clustered) — variante 2: janelas largas ───────────
    ("RC", 2,  25, "low",   542),
    ("RC", 2,  25, "medium",543),
    ("RC", 2,  25, "high",  544),
    ("RC", 2,  50, "low",   552),
    ("RC", 2,  50, "medium",553),
    ("RC", 2,  50, "high",  554),
    ("RC", 2, 100, "low",   562),
    ("RC", 2, 100, "medium",563),
    ("RC", 2, 100, "high",  564),
]


def generate_all_instances(output_dir: str) -> List[Dict]:
    """Gera todas as 54 instancias e salva em JSON."""
    os.makedirs(output_dir, exist_ok=True)
    catalog = []

    print(f"Gerando {len(INSTANCE_PLAN)} instancias CCVRPTW-ST...")
    for i, (fam, var, size, unc, seed) in enumerate(INSTANCE_PLAN, 1):
        inst = generate_instance(fam, size, unc, seed, variant=var)
        path = os.path.join(output_dir, f"{inst.name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(inst), f, indent=2, ensure_ascii=False)

        # Estatisticas da instancia
        demands = [c["demand_kg"] for c in inst.customers]
        values  = [c["load_value_BRL"] for c in inst.customers]
        cats    = [c["product_category"] for c in inst.customers]
        total_val = sum(values)

        print(f"  [{i:02d}/{len(INSTANCE_PLAN)}] {inst.name:45s} | "
              f"n={size:3d} | demanda_total={sum(demands):7.0f} kg | "
              f"valor_carga=R${total_val:,.0f}")

        catalog.append({
            "name": inst.name,
            "family": inst.family,
            "size": size,
            "uncertainty": unc,
            "seed": seed,
            "num_customers": size,
            "num_vehicles": inst.fleet["num_vehicles"],
            "capacity_kg": inst.fleet["capacity_kg"],
            "total_demand_kg": round(sum(demands), 1),
            "total_cargo_value_BRL": round(total_val, 2),
            "sigma_T_C": inst.thermal_profile["sigma_T_C"],
            "dominant_category": inst.thermal_profile["dominant_category"],
            "tw_width_min": inst.generation_metadata["tw_width_min"],
            "file": f"{inst.name}.json",
        })

    return catalog


def save_catalog(catalog: List[Dict], output_dir: str):
    """Salva o catalogo em CSV e JSON."""
    # CSV
    csv_path = os.path.join(output_dir, "instance_catalog.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=catalog[0].keys())
        writer.writeheader()
        writer.writerows(catalog)

    # JSON
    json_path = os.path.join(output_dir, "instance_catalog.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nCatalogo salvo em:\n  {csv_path}\n  {json_path}")


def print_summary(catalog: List[Dict]):
    """Imprime estatisticas do conjunto de instancias."""
    print("\n" + "="*70)
    print("RESUMO DO CONJUNTO DE INSTANCIAS CCVRPTW-ST")
    print("="*70)
    print(f"  Total de instancias: {len(catalog)}")
    sizes      = sorted(set(c["size"] for c in catalog))
    families   = sorted(set(c["family"] for c in catalog))
    variants   = sorted(set(c["family"][-1] for c in catalog))
    uncertainties = ["low", "medium", "high"]
    print(f"  Familias:    {', '.join(families)}")
    print(f"  Variantes:   tipo 1 (janelas estreitas) e tipo 2 (janelas largas)")
    print(f"  Tamanhos:    {', '.join(str(s) for s in sizes)} clientes")
    print(f"  Incertezas:  baixa (0.5C), media (1.0C), alta (2.0C)")
    print()
    print("  Por tamanho:")
    for s in sizes:
        sub = [c for c in catalog if c["size"] == s]
        avg_dem = sum(c["total_demand_kg"] for c in sub) / len(sub)
        print(f"    n={s:3d}: {len(sub):2d} instancias | demanda media={avg_dem:,.0f} kg")
    print()
    print("  Por familia e variante:")
    for fam in families:
        sub = [c for c in catalog if c["family"] == fam]
        print(f"    {fam}: {len(sub):2d} instancias")
    print()
    print("  Por nivel de incerteza:")
    for u in uncertainties:
        sub = [c for c in catalog if c["uncertainty"] == u]
        sigma = sub[0]["sigma_T_C"] if sub else "-"
        print(f"    {u:8s}: {len(sub):2d} instancias | sigma_T = {sigma} C")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCOES AUXILIARES PARA USO EXTERNO
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: str) -> dict:
    """Carrega uma instancia JSON do disco."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_distance_matrix(instance: dict) -> List[List[float]]:
    """
    Calcula a matriz de distancias euclidianas (km) entre deposito e clientes.
    Indice 0 = deposito; indices 1..n = clientes.
    """
    nodes = [instance["depot"]] + instance["customers"]
    n = len(nodes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = round(
                    euclidean(nodes[i]["x"], nodes[i]["y"],
                              nodes[j]["x"], nodes[j]["y"]), 4
                )
    return mat


def compute_travel_time_matrix(instance: dict) -> List[List[float]]:
    """
    Calcula a matriz de tempos de percurso nominais (minutos).
    """
    speed = instance["fleet"]["speed_km_h"]
    nodes = [instance["depot"]] + instance["customers"]
    n = len(nodes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = euclidean(nodes[i]["x"], nodes[i]["y"],
                                 nodes[j]["x"], nodes[j]["y"])
                mat[i][j] = round((dist / speed) * 60.0, 4)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "instances", "generated")
    OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)

    catalog = generate_all_instances(OUTPUT_DIR)
    save_catalog(catalog, OUTPUT_DIR)
    print_summary(catalog)n={s:3d}: {len(sub):2d} instancias | demanda media={avg_dem:,.0f} kg")
    print()
    print("  Por familia e variante:")
    for fam in families:
        sub = [c for c in catalog if c["family"] == fam]
        print(f"    {fam}: {len(sub):2d} instancias")
    print()
    print("  Por nivel de incerteza:")
    for u in uncertainties:
        sub = [c for c in catalog if c["uncertainty"] == u]
        sigma = sub[0]["sigma_T_C"] if sub else "-"
        print(f"    {u:8s}: {len(sub):2d} instancias | sigma_T = {sigma} C")
    print("="*70)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCOES AUXILIARES PARA USO EXTERNO
# ─────────────────────────────────────────────────────────────────────────────

def load_instance(path: str) -> dict:
    """Carrega uma instancia JSON do disco."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_distance_matrix(instance: dict) -> List[List[float]]:
    """
    Calcula a matriz de distancias euclidianas (km) entre deposito e clientes.
    Indice 0 = deposito; indices 1..n = clientes.
    """
    nodes = [instance["depot"]] + instance["customers"]
    n = len(nodes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = round(
                    euclidean(nodes[i]["x"], nodes[i]["y"],
                              nodes[j]["x"], nodes[j]["y"]), 4
                )
    return mat


def compute_travel_time_matrix(instance: dict) -> List[List[float]]:
    """
    Calcula a matriz de tempos de percurso nominais (minutos).
    """
    speed = instance["fleet"]["speed_km_h"]
    nodes = [instance["depot"]] + instance["customers"]
    n = len(nodes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = euclidean(nodes[i]["x"], nodes[i]["y"],
                                 nodes[j]["x"], nodes[j]["y"])
                mat[i][j] = round((dist / speed) * 60.0, 4)
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "instances", "generated")
    OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)

    catalog = generate_all_instances(OUTPUT_DIR)
    save_catalog(catalog, OUTPUT_DIR)
    print_summary(catalog)
