"""Summarise experiment history: tables, significance tests, plots and every number quoted in the report.

Outputs (in --outdir):
    summary.csv            final result per variant (Table 2 of the report)
    tests.csv              Mann-Whitney U tests with effect sizes
    target_distances.csv   mean distance of the final best bodies to every target
    report_numbers.txt     all numbers that are mentioned in the text of the report, in one place
    report_figure.pdf      the two-panel figure of the report
    convergence.png, body_size.png, final_boxplot.png   the same plots as separate images
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).parent

#fixed order and colours so every figure in the report looks the same.
ORDER = ["point+crossover", "point", "subtree+crossover", "subtree", "random_search"]
COLOURS = {"point+crossover": "tab:blue", "point": "tab:cyan", "subtree+crossover": "tab:green",
           "subtree": "tab:olive", "random_search": "tab:gray"}
#the comparisons on the final best fitness: crossover on vs off, every EA vs the baseline,
#and the two mutation operators when crossover is used.
COMPARISONS = [("point+crossover", "point"), ("subtree+crossover", "subtree"),
               ("point+crossover", "random_search"), ("point", "random_search"),
               ("subtree+crossover", "random_search"), ("subtree", "random_search"),
               ("point+crossover", "subtree+crossover")]


def per_generation(rows: list[dict], variant: str, key: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return generations, and the mean and sample std over the independent runs for one logged value."""
    variant_rows = [r for r in rows if r["variant"] == variant and key in r]
    generations = sorted({r["generation"] for r in variant_rows})
    values = [[r[key] for r in variant_rows if r["generation"] == g] for g in generations]
    return generations, np.array([np.mean(v) for v in values]), np.array([np.std(v, ddof=1) for v in values])


def line_plot(rows: list[dict], variants: list[str], key: str, ylabel: str, path: Path) -> None:
    """Plot mean +/- std over the independent runs against the generation number."""
    plt.figure(figsize=(7, 4.2))
    for variant in variants:
        generations, mean, std = per_generation(rows, variant, key)
        plt.plot(generations, mean, label=variant, color=COLOURS.get(variant))
        plt.fill_between(generations, mean - std, mean + std, alpha=0.15, color=COLOURS.get(variant))
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def report_figure(rows: list[dict], variants: list[str], path: Path) -> None:
    """Two-panel vector figure sized for the full text width of the two-column report."""
    labels = {"point+crossover": "point + crossover", "point": "point, no crossover",
              "subtree+crossover": "subtree + crossover", "subtree": "subtree, no crossover",
              "random_search": "random search"}
    styles = {"point+crossover": "-", "point": "--", "subtree+crossover": "-", "subtree": "--", "random_search": ":"}
    colours = {"point+crossover": "tab:blue", "point": "tab:blue", "subtree+crossover": "tab:orange",
               "subtree": "tab:orange", "random_search": "black"}
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    panels = [("best", "Best fitness (lower is better)", variants),
              ("mean_modules", "Mean body size (nodes)", [v for v in variants if v != "random_search"])]
    for ax, (key, ylabel, shown) in zip(axes, panels):
        for variant in shown:
            generations, mean, std = per_generation(rows, variant, key)
            ax.plot(generations, mean, styles[variant], color=colours[variant], label=labels[variant], linewidth=1.2)
            ax.fill_between(generations, mean - std, mean + std, color=colours[variant], alpha=0.10, linewidth=0)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, max(generations))
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].legend(ncol=2, frameon=False, loc="upper right")
    axes[0].set_title("(a)", loc="left", fontsize=8)
    axes[1].set_title("(b)", loc="left", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)
    plt.rcdefaults()


def runs_of(rows: list[dict], variant: str) -> dict[int, list[dict]]:
    """Group the rows of one variant per seed, sorted by generation."""
    runs: dict[int, list[dict]] = {}
    for r in rows:
        if r["variant"] == variant:
            runs.setdefault(r["seed"], []).append(r)
    return {seed: sorted(run, key=lambda r: r["generation"]) for seed, run in sorted(runs.items())}


def compare(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Two-sided Mann-Whitney U test. A12 is the probability that a value of a is larger (worse) than one of b."""
    test = mannwhitneyu(a, b, alternative="two-sided")
    return float(test.statistic), float(test.pvalue), float(test.statistic) / (len(a) * len(b))


def target_numbers(target_dir: Path, samples_per_size: int = 150) -> list[str]:
    """Numbers about the target set itself that are used in the report to give the fitness values a scale."""
    from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
    from ariel.ec.genotypes.tree.operators import random_tree
    from tree_edit_distance import mean_plus_std_tree_edit_distance, tree_edit_distance

    targets = [load_graph_from_json(p) for p in sorted(target_dir.glob("*.json"))]
    sizes = [t.number_of_nodes() for t in targets]
    pairwise = [tree_edit_distance(a, b) for a, b in itertools.combinations(targets, 2)]
    own = [mean_plus_std_tree_edit_distance(t, targets) for t in targets]
    lines = [f"target sizes (nodes): {sizes}",
             f"pairwise target distances: min {min(pairwise):.1f}, max {max(pairwise):.1f}, mean {np.mean(pairwise):.2f}",
             f"fitness of each target when used as a solution: {[round(f, 2) for f in own]} (best {min(own):.2f})"]

    #size-based reference value: every missing or extra module costs one edit, so d_i >= |n - s_i|.
    #f = mean + std is not monotone in the distances, so we minimise f over ALL distance vectors that
    #satisfy these constraints (in steps of 0.5, the smallest edit cost), for every body size n.
    best_value, best_n = float("inf"), []
    steps = np.arange(0, 8.5, 0.5)
    for n in range(1, max(sizes) + 1):
        grids = np.meshgrid(*[abs(n - s) + steps for s in sizes], indexing="ij")
        d = np.stack([g.ravel() for g in grids], axis=1)
        value = float((d.mean(axis=1) + d.std(axis=1)).min())
        if value < best_value - 1e-9:
            best_value, best_n = value, [n]
        elif abs(value - best_value) < 1e-9:
            best_n.append(n)
    lines.append(f"size-based reference value (lower bound from body size only): f >= {best_value:.2f}, reached for n = {best_n} nodes")

    #mean fitness of random bodies per size, to check why populations shrink in the first generations.
    random.seed(0)
    by_size = {}
    for modules in range(1, 21):
        fits = [mean_plus_std_tree_edit_distance(random_tree(max_modules=modules).to_networkx(), targets)
                for _ in range(samples_per_size)]
        by_size[modules + 1] = float(np.mean(fits))
    best_sizes = sorted(by_size, key=by_size.get)[:3]
    lines.append(f"mean fitness of {samples_per_size} random bodies per size (nodes: fitness): "
                 + ", ".join(f"{n}: {by_size[n]:.2f}" for n in (3, 6, 9, 12, 15, 18, 21)))
    lines.append(f"random bodies score best on average at {sorted(best_sizes)} nodes")
    return lines


def main() -> None:
    """Turn raw run histories into report-ready statistics, tests and plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path, nargs="+", help="one or more history .jsonl files")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument("--target-dir", type=Path, default=ROOT / "target_bodies")
    parser.add_argument("--no-target-numbers", action="store_true", help="skip the numbers about the target set")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    #read the experiment records of all given files back into memory.
    rows = [json.loads(line) for path in args.history for line in path.read_text().splitlines()]
    variants = [v for v in ORDER if any(r["variant"] == v for r in rows)]
    runs = {v: runs_of(rows, v) for v in variants}
    last = max(r["generation"] for r in rows)
    #the final-generation rows hold the end result of each independent run.
    final = {v: [run[-1] for run in runs[v].values()] for v in variants}
    best = {v: [r["best"] for r in final[v]] for v in variants}
    numbers: list[str] = []

    #table 1: end result per variant (sample std, because the runs are a sample of all possible runs).
    #mu and sigma are the two terms of the fitness (mean and std of the distances to the targets).
    lines = ["variant,runs,mean_final_best,std_final_best,median,min,max,mean_mu,mean_sigma,mean_best_modules"]
    for v in variants:
        f = np.array(best[v])
        dist = np.array([r["best_distances"] for r in final[v]])
        lines.append(f"{v},{len(f)},{f.mean():.3f},{f.std(ddof=1):.3f},{np.median(f):.3f},{f.min():.3f},{f.max():.3f},"
                     f"{dist.mean(axis=1).mean():.3f},{dist.std(axis=1).mean():.3f},"
                     f"{np.mean([r['best_modules'] for r in final[v]]):.1f}")
    (args.outdir / "summary.csv").write_text("\n".join(lines) + "\n")

    #table 2: two-sided Mann-Whitney U tests (no normality assumption) on the final best fitness.
    lines = ["variant_a,variant_b,mean_a,mean_b,U,p_value,A12"]
    n_tests = 0
    for a, b in COMPARISONS:
        if a in best and b in best:
            u, p, a12 = compare(best[a], best[b])
            lines.append(f"{a},{b},{np.mean(best[a]):.3f},{np.mean(best[b]):.3f},{u:.1f},{p:.3e},{a12:.3f}")
            n_tests += 1
    (args.outdir / "tests.csv").write_text("\n".join(lines) + "\n")

    #table 3: mean distance of the final best body to each of the targets.
    n_targets = len(final[variants[0]][0]["best_distances"])
    lines = ["variant," + ",".join(f"target_{i:02d}" for i in range(n_targets))]
    for v in variants:
        dist = np.mean([r["best_distances"] for r in final[v]], axis=0)
        lines.append(v + "," + ",".join(f"{d:.2f}" for d in dist))
    (args.outdir / "target_distances.csv").write_text("\n".join(lines) + "\n")

    #the behaviour during the runs: how early the progress is made and how the body size develops.
    numbers.append("== progress during the runs ==")
    for v in variants:
        fraction = [(run[0]["best"] - run[20]["best"]) / (run[0]["best"] - run[-1]["best"])
                    for run in runs[v].values() if run[0]["best"] > run[-1]["best"] and last >= 20]
        last_improvement = [max([0] + [g for g in range(1, len(run)) if run[g]["best"] < run[g - 1]["best"] - 1e-12])
                            for run in runs[v].values()]
        numbers.append(f"{v}: mean best fitness at generation 0 = {np.mean([run[0]['best'] for run in runs[v].values()]):.2f}, "
                       f"share of total improvement reached at generation 20 = {np.mean(fraction):.0%}, "
                       f"median generation of last improvement = {np.median(last_improvement):.1f}")
    numbers.append("")
    numbers.append("== body size (nodes, core included) ==")
    for v in variants:
        if "mean_modules" in final[v][0]:
            size = {g: np.mean([run[g]["mean_modules"] for run in runs[v].values()]) for g in (0, 2, 10, 50, last) if g <= last}
            numbers.append(f"{v}: mean population body size " + ", ".join(f"gen {g}: {s:.2f}" for g, s in size.items())
                           + f" | mean size of final best body: {np.mean([r['best_modules'] for r in final[v]]):.1f}")
    if "point" in runs and "point+crossover" in runs:
        u, p, a12 = compare([r["mean_modules"] for r in final["point"]], [r["mean_modules"] for r in final["point+crossover"]])
        numbers.append(f"test on mean population body size at generation {last}, point vs point+crossover: U = {u:.1f}, p = {p:.2e}")
        n_tests += 1

    #interaction: is the benefit of crossover larger for point mutation than for subtree mutation?
    #the runs are paired by seed (same initial population), so we use a Wilcoxon signed-rank test.
    if all(v in runs for v in ORDER[:4]):
        gain = {m: np.array([runs[m][s][-1]["best"] - runs[m + "+crossover"][s][-1]["best"] for s in runs[m]])
                for m in ("point", "subtree")}
        test = wilcoxon(gain["point"], gain["subtree"])
        numbers += ["", "== interaction between crossover and mutation operator ==",
                    f"mean benefit of crossover (fitness without - fitness with): point {gain['point'].mean():.3f}, "
                    f"subtree {gain['subtree'].mean():.3f}",
                    f"Wilcoxon signed-rank test on the per-seed benefits: W = {test.statistic:.1f}, p = {test.pvalue:.2e}"]
        n_tests += 1
        #how large could an effect of crossover be for subtree mutation? bootstrap 95% CI of the difference in means.
        rng = np.random.default_rng(0)
        a, b = np.array(best["subtree+crossover"]), np.array(best["subtree"])
        diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
        numbers.append(f"subtree+crossover minus subtree: difference in means {a.mean() - b.mean():+.3f}, "
                       f"bootstrap 95% CI [{np.percentile(diffs, 2.5):+.2f}, {np.percentile(diffs, 97.5):+.2f}]")

    numbers += ["", "== best body found in these runs =="]
    top = min((r for v in variants for r in final[v]), key=lambda r: r["best"])
    numbers.append(f"{top['variant']}, seed {top['seed']}: f = {top['best']:.3f}, {top['best_modules']} nodes, "
                   f"distances to targets {top['best_distances']}")
    numbers += ["", f"number of statistical tests in this analysis: {n_tests}"]

    if not args.no_target_numbers:
        numbers += ["", "== target set =="] + target_numbers(args.target_dir)
    (args.outdir / "report_numbers.txt").write_text("\n".join(numbers) + "\n")

    #figure 1: the required convergence plot. figure 2: body size, to explain the convergence behaviour.
    line_plot(rows, variants, "best", "Best fitness (lower is better)", args.outdir / "convergence.png")
    line_plot(rows, [v for v in variants if v != "random_search"], "mean_modules",
              "Mean body size in population (nodes)", args.outdir / "body_size.png")

    #the two line plots combined into one compact figure for the report.
    report_figure(rows, variants, args.outdir / "report_figure.pdf")

    #figure 3: spread of the end results, one box per variant.
    plt.figure(figsize=(7, 4.2))
    plt.boxplot([best[v] for v in variants], tick_labels=variants)
    plt.ylabel("Final best fitness (lower is better)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(args.outdir / "final_boxplot.png", dpi=200)
    plt.close()

    for name in ("summary.csv", "tests.csv", "target_distances.csv", "report_numbers.txt"):
        print(f"--- {name}\n{(args.outdir / name).read_text()}")


if __name__ == "__main__":
    main()
