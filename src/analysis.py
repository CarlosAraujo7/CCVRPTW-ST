#!/usr/bin/env python3
"""
Analysis — CCVRPTW-ST
======================
Agrega métricas de todos os runs, executa testes estatísticos
e gera tabelas de resultados prontas para LaTeX/CSV.

Uso:
  python analysis.py                    # processa todos os resultados
  python analysis.py --output analysis  # pasta de saída
  python analysis.py --latex            # gera tabelas .tex também
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.normpath(os.path.join(SRC_DIR, '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'results', 'analysis')

METRICS = ['hypervolume', 'spread', 'best_cost_BRL', 'best_quality_pct', 'best_co2_kg']
METRIC_LABELS = {
    'hypervolume':      'HV',
    'spread':           'Spread (Δ)',
    'best_cost_BRL':    'Custo mín. (R$)',
    'best_quality_pct': 'Qualid. máx. (%)',
    'best_co2_kg':      'CO₂ mín. (kg)',
}
# Para HV e qualidade: maior é melhor. Para custo e CO₂: menor é melhor.
HIGHER_BETTER = {'hypervolume', 'best_quality_pct'}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARREGAMENTO DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_metrics(results_dir: str) -> List[dict]:
    """Carrega todos os arquivos *_metrics.json do diretório de resultados."""
    pattern = os.path.join(results_dir, '**', '*_metrics.json')
    files   = sorted(glob.glob(pattern, recursive=True))
    records = []
    for fpath in files:
        try:
            with open(fpath, encoding='utf-8') as f:
                records.append(json.load(f))
        except Exception as e:
            print(f"  AVISO: não foi possível ler {fpath}: {e}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. AGREGAÇÃO POR INSTÂNCIA × ALGORITMO
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(records: List[dict]) -> Dict[Tuple, dict]:
    """
    Agrega múltiplas seeds por (instância, algoritmo).
    Retorna dicionário: {(instance, algorithm): {metric: [values...]}}
    """
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in records:
        key = (r['instance'], r['algorithm'])
        groups[key].append(r)

    agg = {}
    for key, runs in groups.items():
        entry = {
            'instance':    key[0],
            'algorithm':   key[1],
            'family':      runs[0].get('family', ''),
            'size':        runs[0].get('size', 0),
            'uncertainty': runs[0].get('uncertainty', ''),
            'sigma_T_C':   runs[0].get('sigma_T_C', 0),
            'n_runs':      len(runs),
            'elapsed_s_mean': round(np.mean([r.get('elapsed_s', 0) for r in runs]), 1),
        }
        for m in METRICS:
            vals = [r[m] for r in runs if r.get(m) is not None]
            if vals:
                entry[f'{m}_mean'] = round(float(np.mean(vals)), 6)
                entry[f'{m}_std']  = round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0), 6)
                entry[f'{m}_best'] = round(float(max(vals) if m in HIGHER_BETTER else min(vals)), 6)
            else:
                entry[f'{m}_mean'] = None
                entry[f'{m}_std']  = None
                entry[f'{m}_best'] = None
        agg[key] = entry
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 3. TESTE DE WILCOXON SIGNED-RANK
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_signed_rank(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Teste de Wilcoxon signed-rank para amostras emparelhadas.
    Retorna (estatística W, p-valor aproximado via distribuição normal).
    Ref.: Wilcoxon (1945); Demšar (2006) JMLR 7:1-30.
    """
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = sorted(enumerate(diffs), key=lambda t: abs(t[1]))
    ranks = [0.0] * n

    # Atribui ranks (com ties médios)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs(abs_diffs[j][1]) == abs(abs_diffs[j+1][1]):
            j += 1
        rank_avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[abs_diffs[k][0]] = rank_avg
        i = j + 1

    W_plus  = sum(ranks[abs_diffs[i][0]] for i in range(n) if diffs[abs_diffs[i][0]] > 0)
    W       = min(W_plus, n*(n+1)/2 - W_plus)

    # Aproximação normal (válida para n ≥ 10)
    mu      = n * (n + 1) / 4.0
    sigma   = math.sqrt(n * (n + 1) * (2*n + 1) / 24.0)
    z       = (W - mu) / sigma if sigma > 0 else 0.0
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return W, round(p_value, 6)


def _norm_cdf(x: float) -> float:
    """CDF da normal padrão (aproximação de Abramowitz & Stegun)."""
    t = 1 / (1 + 0.2316419 * abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    approx = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2) * poly
    return approx if x >= 0 else 1 - approx


def run_statistical_tests(
    records:    List[dict],
    algorithms: List[str],
    metric:     str = 'hypervolume',
) -> List[dict]:
    """
    Executa Wilcoxon signed-rank entre todos os pares de algoritmos
    para a métrica especificada (valores emparelhados por instância+seed).
    """
    # Agrupa valores por (instância, seed, algoritmo)
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get(metric) is not None:
            data[r['instance']][r['algorithm']].append(r[metric])

    results = []
    for (a1, a2) in combinations(algorithms, 2):
        x_vals, y_vals = [], []
        for inst, algo_data in data.items():
            if a1 in algo_data and a2 in algo_data:
                # Emparelha por posição (seed 0..N)
                vals1 = algo_data[a1]
                vals2 = algo_data[a2]
                n = min(len(vals1), len(vals2))
                x_vals.extend(vals1[:n])
                y_vals.extend(vals2[:n])

        if len(x_vals) < 5:
            continue

        W, p = wilcoxon_signed_rank(x_vals, y_vals)
        mean1, mean2 = np.mean(x_vals), np.mean(y_vals)
        winner = a1 if (mean1 > mean2) == (metric in HIGHER_BETTER) else a2
        results.append({
            'metric':       metric,
            'algo_A':       a1,
            'algo_B':       a2,
            'mean_A':       round(float(mean1), 6),
            'mean_B':       round(float(mean2), 6),
            'winner':       winner,
            'W_statistic':  round(W, 2),
            'p_value':      p,
            'significant':  p < 0.05,
            'n_pairs':      len(x_vals),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. GERAÇÃO DE TABELAS
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(agg: Dict, output_dir: str):
    """Tabela mestra: uma linha por (instância, algoritmo) com todas as métricas."""
    path = os.path.join(output_dir, 'summary.csv')
    rows = sorted(agg.values(), key=lambda r: (r['size'], r['family'], r['uncertainty'], r['algorithm']))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  → {path}")


def save_hv_comparison_csv(agg: Dict, algorithms: List[str], output_dir: str):
    """
    Tabela de comparação de HV: linhas = instâncias, colunas = algoritmos.
    Formato: inst | HV_algo1 ± std | HV_algo2 ± std | ...
    """
    path = os.path.join(output_dir, 'hv_comparison.csv')
    instances = sorted({k[0] for k in agg})

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        header = ['instance', 'family', 'size', 'uncertainty'] + algorithms
        w = csv.writer(fh)
        w.writerow(header)
        for inst in instances:
            # Metadados da instância
            meta = next((agg[(inst, a)] for a in algorithms if (inst, a) in agg), {})
            row = [inst, meta.get('family',''), meta.get('size',''), meta.get('uncertainty','')]
            for algo in algorithms:
                entry = agg.get((inst, algo))
                if entry and entry.get('hypervolume_mean') is not None:
                    row.append(f"{entry['hypervolume_mean']:.4e} ± {entry['hypervolume_std']:.2e}")
                else:
                    row.append('—')
            w.writerow(row)
    print(f"  → {path}")


def save_wilcoxon_csv(tests: List[dict], output_dir: str, metric: str):
    """Resultados dos testes de Wilcoxon."""
    path = os.path.join(output_dir, f'wilcoxon_{metric}.csv')
    if not tests:
        return
    fieldnames = list(tests[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(tests)
    print(f"  → {path}")


def save_by_size_csv(agg: Dict, algorithms: List[str], metric: str, output_dir: str):
    """Média da métrica por tamanho de instância e algoritmo."""
    path = os.path.join(output_dir, f'{metric}_by_size.csv')
    size_data: Dict[Tuple, List[float]] = defaultdict(list)
    for (inst, algo), entry in agg.items():
        sz  = entry['size']
        val = entry.get(f'{metric}_mean')
        if val is not None:
            size_data[(sz, algo)].append(val)

    sizes = sorted({entry['size'] for entry in agg.values()})
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['size'] + algorithms)
        for sz in sizes:
            row = [sz]
            for algo in algorithms:
                vals = size_data.get((sz, algo), [])
                row.append(f"{np.mean(vals):.4e}" if vals else '—')
            w.writerow(row)
    print(f"  → {path}")


def save_by_uncertainty_csv(agg: Dict, algorithms: List[str], metric: str, output_dir: str):
    """Média da métrica por nível de incerteza e algoritmo."""
    path = os.path.join(output_dir, f'{metric}_by_uncertainty.csv')
    unc_data: Dict[Tuple, List[float]] = defaultdict(list)
    for (inst, algo), entry in agg.items():
        unc = entry['uncertainty']
        val = entry.get(f'{metric}_mean')
        if val is not None:
            unc_data[(unc, algo)].append(val)

    uncertainties = ['LOW', 'MEDIUM', 'HIGH']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['uncertainty', 'sigma_T_C'] + algorithms)
        sigma_map = {'LOW': 0.5, 'MEDIUM': 1.0, 'HIGH': 2.0}
        for unc in uncertainties:
            row = [unc, sigma_map[unc]]
            for algo in algorithms:
                vals = unc_data.get((unc, algo), [])
                row.append(f"{np.mean(vals):.4e}" if vals else '—')
            w.writerow(row)
    print(f"  → {path}")


def save_latex_table(agg: Dict, algorithms: List[str], output_dir: str):
    """
    Tabela LaTeX compacta: HV médio por família × tamanho × algoritmo.
    Melhor valor em negrito.
    """
    path = os.path.join(output_dir, 'table_hv_latex.tex')
    families = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
    sizes    = [25, 50, 100]

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Hipervolume médio (HV) $\pm$ desvio padrão por instância e algoritmo}',
        r'\label{tab:hv_results}',
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{ll' + 'r' * len(algorithms) + '}',
        r'\toprule',
        r'Família & $n$' + ''.join(f' & {a}' for a in algorithms) + r' \\',
        r'\midrule',
    ]

    for fam in families:
        first = True
        for sz in sizes:
            row_vals = {}
            for algo in algorithms:
                # Média sobre todas as instâncias desta família+tamanho
                vals = [
                    entry['hypervolume_mean']
                    for (inst, alg), entry in agg.items()
                    if alg == algo
                    and entry['family'] == fam
                    and entry['size'] == sz
                    and entry.get('hypervolume_mean') is not None
                ]
                row_vals[algo] = (np.mean(vals), np.std(vals)) if vals else (None, None)

            best_val = max((v for v, _ in row_vals.values() if v is not None), default=None)
            fam_str  = fam if first else ''
            first    = False

            cells = []
            for algo in algorithms:
                v, s = row_vals[algo]
                if v is None:
                    cells.append('—')
                else:
                    txt = f'{v:.3e}'
                    if best_val is not None and abs(v - best_val) < 1e-10:
                        txt = r'\textbf{' + txt + '}'
                    cells.append(txt)

            lines.append(f'{fam_str} & {sz}' + ''.join(f' & {c}' for c in cells) + r' \\')
        lines.append(r'\midrule')

    lines += [
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}',
    ]

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. RESUMO NO TERMINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(agg: Dict, algorithms: List[str]):
    """Imprime tabela resumida de HV por algoritmo no terminal."""
    print(f"\n{'='*70}")
    print(f"  RESUMO — Hipervolume médio por algoritmo e tamanho")
    print(f"{'='*70}")

    for sz in [25, 50, 100]:
        print(f"\n  n = {sz}")
        print(f"  {'Algoritmo':<15}" + ''.join(f"{'HV (média±std)':>20}"))
        header = f"  {'Algoritmo':<15}" + ''.join(f"{a:>20}" for a in algorithms)
        print(header)
        print(f"  {'-'*15}" + ''.join(f"  {'─'*17}" for _ in algorithms))

        uncerts = ['LOW', 'MEDIUM', 'HIGH']
        for unc in uncerts:
            row = f"  σ={unc:<10}"
            for algo in algorithms:
                vals = [
                    entry['hypervolume_mean']
                    for (inst, alg), entry in agg.items()
                    if alg == algo and entry['size'] == sz
                    and entry['uncertainty'] == unc
                    and entry.get('hypervolume_mean') is not None
                ]
                if vals:
                    row += f"  {np.mean(vals):.3e} ± {np.std(vals):.1e}"
                else:
                    row += f"  {'—':>18}"
            print(row)

    print(f"\n{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Análise de resultados CCVRPTW-ST')
    parser.add_argument('--results-dir', default=RESULTS_DIR,
                        help='Diretório com resultados (padrão: ../results)')
    parser.add_argument('--output',      default=ANALYSIS_DIR,
                        help='Diretório de saída da análise')
    parser.add_argument('--metric',      default='hypervolume',
                        choices=METRICS,
                        help='Métrica principal para testes estatísticos')
    parser.add_argument('--latex',       action='store_true',
                        help='Gera tabelas LaTeX além dos CSVs')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Carregando resultados de: {args.results_dir}")
    records = load_all_metrics(args.results_dir)
    if not records:
        print("Nenhum resultado encontrado. Execute experiment_runner.py primeiro.")
        return

    algorithms = sorted({r['algorithm'] for r in records})
    print(f"Registros carregados: {len(records)}")
    print(f"Algoritmos: {', '.join(algorithms)}")
    print(f"Instâncias únicas: {len({r['instance'] for r in records})}")

    print(f"\nAgregando métricas por (instância, algoritmo)...")
    agg = aggregate(records)

    print(f"\nSalvando tabelas em: {args.output}")
    save_summary_csv(agg, args.output)
    save_hv_comparison_csv(agg, algorithms, args.output)
    save_by_size_csv(agg, algorithms, 'hypervolume', args.output)
    save_by_uncertainty_csv(agg, algorithms, 'hypervolume', args.output)

    print(f"\nExecutando testes de Wilcoxon ({args.metric})...")
    tests = run_statistical_tests(records, algorithms, metric=args.metric)
    save_wilcoxon_csv(tests, args.output, args.metric)

    if len(algorithms) > 1 and tests:
        print(f"\n  Pares testados (p < 0.05 = diferença significativa):")
        for t in tests:
            sig = '✓' if t['significant'] else '✗'
            print(f"  {sig} {t['algo_A']:12s} vs {t['algo_B']:12s}: "
                  f"p={t['p_value']:.4f}  vencedor={t['winner']}")

    if args.latex:
        print(f"\nGerando tabela LaTeX...")
        save_latex_table(agg, algorithms, args.output)

    print_summary(agg, algorithms)
    print(f"\nAnálise salva em: {args.output}")


if __name__ == '__main__':
    main()
