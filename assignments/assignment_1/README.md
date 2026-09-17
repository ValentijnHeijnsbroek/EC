# Assignment 1 – crossover and mutation for evolving robot bodies

Research question: does subtree crossover improve the EA, and does this depend on whether
the mutation operator can add modules?

## Files

| File | What it does |
|---|---|
| `evolve_assignment.py` | The EA (4 variants) and the random search baseline. Writes one row per generation to a `.jsonl` file. |
| `analyse_results.py` | Makes all tables, statistical tests, figures and `report_numbers.txt` (every number that is mentioned in the text of the report). |
| `A1_template_2026.py`, `tree_edit_distance.py`, `target_bodies/` | Provided by the course, not changed. |
| `results/final_30seeds/` | Main experiment of the report (seeds 1–30). |
| `results/robustness_pm02_10seeds/` | Control experiment with mutation probability 0.2 (seeds 1–10). |
| `results/first_tryout/` | Our first try-out that is mentioned in the introduction. It was made with an earlier version of the script (point vs. subtree mutation, always with crossover, mutation probability 0.2, all initial bodies of maximum size, no module cap, seeds 11–55). Kept for reference only. |

Nothing in `src/ariel` was changed. From ARIEL we use the tree genome and its operators
(`random_tree`, `crossover_subtree`, `mutate_replace_node`, `mutate_subtree_replacement`).
The EA loop itself (selection, elitism, budget, logging) is our own code in `evolve_assignment.py`.

## Variants

| Name | Mutation | Crossover probability |
|---|---|---|
| `point+crossover` | point (`mutate_replace_node`) | 0.8 |
| `point` | point | 0.0 |
| `subtree+crossover` | subtree (`mutate_subtree_replacement`) | 0.8 |
| `subtree` | subtree | 0.0 |
| `random_search` | – | – |

Fixed settings: population 50, 100 generations, tournament size 3, 1 elite, mutation probability 1.0,
at most 20 modules, 4950 evaluations per run for every method.

## Reproduce the results

Run from the root of the repository (after `uv venv` and `uv sync`).

Main experiment (5 methods x 30 seeds, about 25 minutes on one core; seeds 1–30 are the default):

```bash
uv run assignments/assignment_1/evolve_assignment.py --output assignments/assignment_1/results/final_30seeds/history.jsonl
uv run assignments/assignment_1/analyse_results.py assignments/assignment_1/results/final_30seeds/history.jsonl --outdir assignments/assignment_1/results/final_30seeds
```

Control experiment with mutation probability 0.2:

```bash
uv run assignments/assignment_1/evolve_assignment.py --variants point+crossover point subtree+crossover subtree --mutation-probability 0.2 --seeds 1 2 3 4 5 6 7 8 9 10 --output assignments/assignment_1/results/robustness_pm02_10seeds/history.jsonl
uv run assignments/assignment_1/analyse_results.py assignments/assignment_1/results/robustness_pm02_10seeds/history.jsonl --outdir assignments/assignment_1/results/robustness_pm02_10seeds --no-target-numbers
```

The runs are deterministic for a given seed, so the same commands give exactly the same numbers.
