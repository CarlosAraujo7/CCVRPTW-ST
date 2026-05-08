#!/usr/bin/env python3
"""
Experiment Runner — CCVRPTW-ST
===============================
Executa o benchmark completo com múltiplos algoritmos, instâncias e sementes
usando execução paralela via multiprocessing.

Algoritmos disponíveis:
  NSGA2-MC   — NSGA-II + Monte Carlo embarcado (mc=30)  [proposto]
  NSGA2-DET  — NSGA-II com avaliação determinística     [baseline]
  NSGA2-MC10 — NSGA-II + Monte Carlo leve (mc=10)       [sensibilidade]
  MOEA-RS    — Busca aleatória multiobjetivo + MC        [baseline inferior]

Exemplos de uso:
  # Execução completa (54 instâncias × 3 algos × 5 seeds):
  python experiment_runner.py --workers 8

  # Teste rápido (n=25, 2 seeds, 50 gerações):
  python experiment_runner.py --quick --workers 4

  # Só instâncias C1 e R1, apenas NSGA2-MC:
  python experiment_runner.py --families C1 R1 --algorithms NSGA2-MC --seeds 5

  # Retomar execução (pula runs já concluídas):
  python experiment_runner.py --workers 8 --resume
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.normpath(os.path.join(SRC_DIR, '..'))
INST_DIR    = os.path.join(BASE_DIR, 'instances', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Parâmetros padrão
DEFAULT_POP_SIZE   = 100
DEFAULT_N_GEN      = 200
DEFAULT_MC_SAMPLES = 30
DEFAULT_SEEDS      = 5
DEFAULT_WORKERS    = 8

# Definição dos algoritmos
ALGO_CONFIG: Dict[str, dict] = {
    'NSGA2-MC': {
        'runner':     'nsga2',
        'mc_samples': 30,
        'label':      'NSGA-II + Monte Carlo (proposto)',
    },
    'NSGA2-DET': {
        'runner':     'nsga2',
        'mc_samples': 0,        # 0 = modo determinístico (T = T_set)
        'label':      'NSGA-II Determinístico (baseline)',
    },
    'NSGA2-MC10': {
        'runner':     'nsga2',
        'mc_samples': 10,
        'label':      'NSGA-II + Monte Carlo leve (sensibilidade)',
    },
    'MOEA-RS': {
        'runner':     'random',
        'mc_samples': 30,
        'label':      'Busca Aleatória Multiobjetivo (baseline inferior)',
    },
}

ALL_FAMILIES = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
ALL_SIZES    = [25, 50, 100]
ALL_UNCERTS  = ['LOW', 'MEDIUM', 'HIGH']


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (executada em cada processo filho)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    """
    Executa uma única combinação (instância, algoritmo, seed) e salva resultados.
    Retorna dicionário de status para relatório do processo pai.
    """
    inst_path  = task['inst_path']
    algo_name  = task['algo']
    seed       = task['seed']
    pop_size   = task['pop_size']
    n_gen      = task['n_gen']
    out_dir    = task['out_dir']
    mc_samples = task['mc_samples']
    runner     = task['runner']

    inst_name  = os.path.basename(inst_path).replace('.json', '')
    result_key = f"{inst_name}__{algo_name}__seed{seed}"

    t0 = time.time()
    try:
        # Importa o módulo localmente no processo filho (evita problemas de pickle)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'nsga2_simheuristic',
            os.path.join(SRC_DIR, 'nsga2_simheuristic.py'),
        )
        ns = importlib.util.module_from_spec(spec)
        sys.modules['nsga2_simheuristic'] = ns
        spec.loader.exec_module(ns)

        with open(inst_path, encoding='utf-8') as f:
            inst = json.load(f)

        # Executa o algoritmo
        if runner == 'nsga2':
            pareto, hv_hist = ns.run_nsga2(
                inst,
                pop_size=pop_size,
                n_gen=n_gen,
                mc_samples=mc_samples,
                seed=seed,
                verbose=False,
            )
        else:  # random search
            pareto, hv_hist = ns.run_random_search(
                inst,
                pop_size=pop_size,
                n_gen=n_gen,
                mc_samples=mc_samples,
                seed=seed,
                verbose=False,
            )

        # Métricas
        hv     = ns.compute_hypervolume(pareto)
        spread = ns.compute_spread(pareto)
        gd     = ns.compute_gd(pareto, pareto)

        metrics = {
            'instance':          inst_name,
            'family':            inst.get('family', ''),
            'size':              inst.get('size', 0),
            'uncertainty':       inst.get('uncertainty_level', ''),
            'sigma_T_C':         inst['thermal_profile']['sigma_T_C'],
            'algorithm':         algo_name,
            'seed':              seed,
            'pop_size':          pop_size,
            'n_gen':             n_gen,
            'mc_samples':        mc_samples,
            'elapsed_s':         round(time.time() - t0, 2),
            'pareto_size':       len(pareto),
            'n_feasible':        sum(1 for x in pareto if x.feasible),
            'hypervolume':       round(hv, 8),
            'spread':            round(spread, 6),
            'gd':                round(gd, 6),
            'best_cost_BRL':     round(min(x.f_cost  for x in pareto), 2) if pareto else None,
            'best_quality_pct':  round((1 - min(x.f_qloss for x in pareto)) * 100, 3) if pareto else None,
            'best_co2_kg':       round(min(x.f_co2   for x in pareto), 4) if pareto else None,
        }

        # Salva resultados
        os.makedirs(out_dir, exist_ok=True)
        prefix = os.path.join(out_dir, f"{result_key}")

        # Pareto front CSV
        import csv
        with open(f"{prefix}_pareto.csv", 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['sol_id', 'f1_cost_BRL', 'f2_quality_loss',
                        'f3_co2_kg', 'n_routes', 'feasible'])
            for i, ind in enumerate(sorted(pareto, key=lambda x: x.f_cost)):
                w.writerow([i+1, round(ind.f_cost,2), round(ind.f_qloss,6),
                             round(ind.f_co2,4), len(ind.routes), ind.feasible])

        # HV history CSV
        with open(f"{prefix}_hv.csv", 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['generation', 'hypervolume'])
            for gen, hv_val in hv_hist:
                w.writerow([gen, round(hv_val, 8)])

        # Metrics JSON
        with open(f"{prefix}_metrics.json", 'w', encoding='utf-8') as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)

        return {'key': result_key, 'status': 'ok', 'elapsed': metrics['elapsed_s'],
                'hv': hv, 'pareto_size': len(pareto)}

    except Exception as e:
        tb = traceback.format_exc()
        err_path = os.path.join(out_dir, f"{result_key}_ERROR.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(err_path, 'w') as fh:
            fh.write(tb)
        return {'key': result_key, 'status': 'error', 'error': str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# MONTAGEM DO PLANO DE EXPERIMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def build_task_list(
    families:   List[str],
    sizes:      List[int],
    algorithms: List[str],
    n_seeds:    int,
    pop_size:   int,
    n_gen:      int,
    resume:     bool,
) -> List[dict]:
    """Constrói lista de tarefas (instância × algoritmo × seed)."""
    tasks = []
    inst_files = sorted(glob.glob(os.path.join(INST_DIR, '*.json')))
    inst_files = [f for f in inst_files if 'catalog' not in os.path.basename(f)]

    for fpath in inst_files:
        name  = os.path.basename(fpath).replace('.json', '')
        parts = name.split('_')
        fam   = parts[1]           # C1, R2, etc.
        sz    = int(parts[2])      # 025 → 25
        unc   = parts[3]           # LOW/MEDIUM/HIGH

        if fam not in families:
            continue
        if sz not in sizes:
            continue

        for algo in algorithms:
            cfg = ALGO_CONFIG[algo]
            out_dir = os.path.join(RESULTS_DIR, algo)

            for seed in range(n_seeds):
                result_key = f"{name}__{algo}__seed{seed}"
                # Verifica se já existe (modo resume)
                if resume and os.path.exists(os.path.join(out_dir, f"{result_key}_metrics.json")):
                    continue

                mc = cfg['mc_samples']
                tasks.append({
                    'inst_path':  fpath,
                    'algo':       algo,
                    'seed':       seed,
                    'pop_size':   pop_size,
                    'n_gen':      n_gen,
                    'mc_samples': mc,
                    'runner':     cfg['runner'],
                    'out_dir':    out_dir,
                })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark CCVRPTW-ST — execução paralela',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workers',    type=int,   default=DEFAULT_WORKERS,
                        help=f'Número de processos paralelos (padrão: {DEFAULT_WORKERS})')
    parser.add_argument('--seeds',      type=int,   default=DEFAULT_SEEDS,
                        help=f'Sementes por combinação (padrão: {DEFAULT_SEEDS})')
    parser.add_argument('--pop-size',   type=int,   default=DEFAULT_POP_SIZE,
                        help=f'Tamanho da população (padrão: {DEFAULT_POP_SIZE})')
    parser.add_argument('--n-gen',      type=int,   default=DEFAULT_N_GEN,
                        help=f'Número de gerações (padrão: {DEFAULT_N_GEN})')
    parser.add_argument('--families',   nargs='+',  default=ALL_FAMILIES,
                        choices=ALL_FAMILIES, metavar='FAM',
                        help='Famílias a executar (padrão: todas)')
    parser.add_argument('--sizes',      nargs='+',  type=int, default=ALL_SIZES,
                        choices=ALL_SIZES, metavar='N',
                        help='Tamanhos de instância (padrão: 25 50 100)')
    parser.add_argument('--algorithms', nargs='+',  default=list(ALGO_CONFIG.keys()),
                        choices=list(ALGO_CONFIG.keys()), metavar='ALGO',
                        help='Algoritmos a executar (padrão: todos)')
    parser.add_argument('--resume',     action='store_true',
                        help='Pula execuções com resultado já salvo')
    parser.add_argument('--quick',      action='store_true',
                        help='Modo rápido: n=25, 2 seeds, 50 gerações, só NSGA2-MC e NSGA2-DET')
    args = parser.parse_args()

    # Modo rápido
    if args.quick:
        args.sizes      = [25]
        args.seeds      = 2
        args.n_gen      = 50
        args.algorithms = ['NSGA2-MC', 'NSGA2-DET']
        print("⚡ Modo rápido: n=25, 2 seeds, 50 gerações, NSGA2-MC + NSGA2-DET")

    tasks = build_task_list(
        families   = args.families,
        sizes      = args.sizes,
        algorithms = args.algorithms,
        n_seeds    = args.seeds,
        pop_size   = args.pop_size,
        n_gen      = args.n_gen,
        resume     = args.resume,
    )

    n_total = len(tasks)
    if n_total == 0:
        print("Nenhuma tarefa pendente. Use --resume=False para forçar reexecução.")
        return

    workers = min(args.workers, n_total, cpu_count())

    print(f"\n{'='*65}")
    print(f"  BENCHMARK CCVRPTW-ST")
    print(f"{'='*65}")
    print(f"  Tarefas    : {n_total}")
    print(f"  Algoritmos : {', '.join(args.algorithms)}")
    print(f"  Famílias   : {', '.join(args.families)}")
    print(f"  Tamanhos   : {args.sizes}")
    print(f"  Seeds      : {args.seeds}")
    print(f"  Pop / Gen  : {args.pop_size} / {args.n_gen}")
    print(f"  Workers    : {workers}")
    print(f"  Resultados : {RESULTS_DIR}")
    print(f"{'='*65}\n")

    t_start = time.time()
    done = 0
    errors = []

    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_worker, tasks, chunksize=1):
            done += 1
            elapsed_total = time.time() - t_start
            avg_s = elapsed_total / done
            eta_s = avg_s * (n_total - done)

            if result['status'] == 'ok':
                status_str = f"HV={result['hv']:.3e}  Pareto={result['pareto_size']}"
            else:
                status_str = f"ERRO: {result['error'][:50]}"
                errors.append(result)

            print(f"  [{done:4d}/{n_total}] {result['key'][:55]:<55}"
                  f" | {result.get('elapsed', 0):.1f}s"
                  f" | ETA {eta_s/60:.1f}min"
                  f" | {status_str}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  Concluído em {elapsed_total/60:.1f} min")
    print(f"  Sucesso: {done - len(errors)}/{done}")
    if errors:
        print(f"  Erros ({len(errors)}): ver arquivos *_ERROR.txt em {RESULTS_DIR}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
