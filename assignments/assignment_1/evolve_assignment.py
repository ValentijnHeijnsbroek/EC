from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np

from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from tree_edit_distance import distances_to_targets, mean_plus_std_tree_edit_distance
from A1_template_2026 import load_targets

ROOT = Path(__file__).parent
TARGET_DIR = ROOT / "target_bodies"

#each EA variant combines a mutation operator with a crossover probability.
#the variants with and without crossover differ in exactly one aspect: the crossover probability.
VARIANTS = {
    "point+crossover": (mutate_replace_node, 0.8),
    "point": (mutate_replace_node, 0.0),
    "subtree+crossover": (mutate_subtree_replacement, 0.8),
    "subtree": (mutate_subtree_replacement, 0.0),
}


def seed_everything(seed: int) -> None:
    """
    This function makes sure that we can reproduce the same random numbers for a seed so we can replicate results for seeds.
    """
    random.seed(seed)
    np.random.seed(seed)


def score(genome: TreeGenome, targets: list[nx.DiGraph]) -> float:
    """This function converts one genome to a body and calculates the fitness.
    """
    body = genome.to_networkx()
    return float(mean_plus_std_tree_edit_distance(body, targets))


def random_body(max_modules: int) -> TreeGenome:
    """Sample one random body with a random size between 1 and max_modules modules.
    random_tree always uses the full budget, so without this every random body has the maximum size.
    """
    return random_tree(max_modules=random.randint(1, max_modules))


def tournament(population: list[tuple[TreeGenome, float]], size: int = 3) -> TreeGenome:
    """Select a parent using tournament selection.
    """
    contestants = random.sample(population, min(size, len(population)))
    return min(contestants, key=lambda item: item[1])[0]


def make_child(
    population: list[tuple[TreeGenome, float]],
    targets: list[nx.DiGraph],
    mutation: Callable[[TreeGenome], None],
    max_modules: int,
    crossover_probability: float,
    mutation_probability: float,
) -> tuple[TreeGenome, float]:
    """Create, mutate, and evaluate one offspring.
    """
    #1: choose two parents from the current population.
    parent_a = tournament(population)
    parent_b = tournament(population)
    #2: crossover combines genetic material from both parents.
    if random.random() < crossover_probability:
        child, _ = crossover_subtree(copy.deepcopy(parent_a), copy.deepcopy(parent_b))
    else:
        #if crossover does not happen, continue with a copy of parent A.
        child = copy.deepcopy(parent_a)
    #3: mutation changes the child and creates new possible solutions.
    if random.random() < mutation_probability:
        if mutation is mutate_subtree_replacement:
            mutation(child, max_modules=max_modules)
        else:
            mutation(child)
    #a child above the module budget is rejected and replaced by a copy of parent A.
    if len(child.nodes) > max_modules + 1:
        child = copy.deepcopy(parent_a)
    #4: score the child before it enters the next population.
    return child, score(child, targets)


def run_ea(
    variant: str,
    seed: int,
    targets: list[nx.DiGraph],
    population_size: int,
    generations: int,
    max_modules: int,
    mutation_probability: float = 1.0,
) -> list[dict[str, object]]:
    """Run one complete evolutionary experiment and return its history.
    Each generation keeps the best individual
    """
    seed_everything(seed)
    #look up which mutation operator and which crossover probability this variant uses.
    mutation, crossover_probability = VARIANTS[variant]
    #the initial population is the first set of candidate bodies.
    population = [
        (genome, score(genome, targets))
        for genome in (random_body(max_modules) for _ in range(population_size))
    ]
    #every fitness calculation counts toward the comparison budget.
    evaluations = len(population)
    history: list[dict[str, object]] = []

    for generation in range(generations + 1):
        #these values show the current population for the convergence plot.
        fitnesses = [fit for _, fit in population]
        best_genome = min(population, key=lambda item: item[1])[0]
        sizes = [len(genome.nodes) for genome, _ in population]
        row = {
            "variant": variant,
            "seed": seed,
            "crossover_probability": crossover_probability,
            "mutation_probability": mutation_probability,
            "generation": generation,
            "evaluations": evaluations,
            "best": min(fitnesses),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "worst": max(fitnesses),
            "best_modules": len(best_genome.nodes),
            "mean_modules": float(np.mean(sizes)),
            "max_modules": max(sizes),
            "best_distances": list(distances_to_targets(best_genome.to_networkx(), targets)),
        }
        #the final best body is saved so it can be shown and analysed in the report.
        if generation == generations:
            row["best_genome"] = best_genome.to_dict()
        history.append(row)
        #do not go over the set number of generations
        if generation == generations:
            break
        #prevent best solution so far from being lost
        elite = min(population, key=lambda item: item[1])
        #all other individuals are newly generated offspring.
        offspring = [make_child(population, targets, mutation, max_modules,
                                crossover_probability, mutation_probability)
                     for _ in range(population_size - 1)]
        evaluations += len(offspring)
        #keep the best one and all the offspring
        population = [elite, *offspring]
        population.sort(key=lambda item: item[1])
    return history


def run_random_search(
    seed: int,
    targets: list[nx.DiGraph],
    population_size: int,
    generations: int,
    max_modules: int,
) -> list[dict[str, object]]:
    """Run random search using exactly the same evaluation budget as the EA.
    Random search is the control condition.
    """
    seed_everything(seed)
    best = float("inf")
    best_genome = None
    evaluations = 0
    history = []
    for generation in range(generations + 1):
        #generation 0 matches the EA's initial population, every later one matches its offspring (the elite is not re-evaluated).
        batch = population_size if generation == 0 else population_size - 1
        for _ in range(batch):
            candidate = random_body(max_modules)
            fitness = score(candidate, targets)
            evaluations += 1
            if fitness < best:
                best, best_genome = fitness, candidate
        row = {
            "variant": "random_search",
            "seed": seed,
            "generation": generation,
            "evaluations": evaluations,
            "best": best,
            "mean": best,
            "std": 0.0,
            "worst": best,
            "best_modules": len(best_genome.nodes),
            "best_distances": list(distances_to_targets(best_genome.to_networkx(), targets)),
        }
        if generation == generations:
            row["best_genome"] = best_genome.to_dict()
        history.append(row)
    return history


def main() -> None:
    """Run all requested variants and write their histories to JSONL.
    """
    #add the variables that can be set from the command line with defaults
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=[*VARIANTS, "random_search"],
                        default=[*VARIANTS, "random_search"])
    parser.add_argument("--mutation-probability", type=float, default=1.0)
    #the experiment in the report uses seeds 1 to 30.
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 31)))
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--max-modules", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "history.jsonl")
    args = parser.parse_args()
    targets = load_targets(TARGET_DIR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for variant in args.variants:
            for seed in args.seeds:
                if variant == "random_search":
                    rows = run_random_search(seed, targets, args.population, args.generations, args.max_modules)
                else:
                    rows = run_ea(variant, seed, targets, args.population, args.generations, args.max_modules,
                                  args.mutation_probability)
                #store the history immediately so long experiments produce usable data.
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
                print(f"completed {variant} seed={seed}")


if __name__ == "__main__":
    main()
