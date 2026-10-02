"""
Hyperparameter study: look for parameters where the populations of preys and predators
oscillate, as in a Lotka-Volterra model.

Each configuration is simulated with several seeds. The populations are then scored by
oscillation_metrics(): both species must survive, oscillate with a clear period and a large
amplitude, the predator peaks must come after the prey peaks, and the populations must not
stay stuck at their maximum (carrying_capacity_prey, max_predators).

The cycles are slow (often several hundred time steps), so the simulations must be long
enough to contain at least 3 of them.

Usage (from the project folder):
    python study.py search --configs 400 --seeds 2 --steps 6000
    python study.py refine --input results/search.csv --top 10 --variants 8 --seeds 4 --steps 8000
    python study.py plot --input results/refine.csv --rank 1 --steps 8000

A long search can be split in several runs with different --rng-seed (the configurations
are then different), and their CSV files given together to --input.

The results are written in results/ as CSV files, one row per simulation, as soon as they are done.
"""
import argparse
import csv
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from world import World

# space of the parameters explored: name -> (low, high, scale)
# "log": sampled uniformly on a log scale, "int": integer
PARAM_SPACE = {
    "size": (80, 140, "int"),                              # width = height of the world
    "carrying_capacity_prey": (150, 500, "int"),
    "n_predators_init": (5, 30, "int"),
    "speed_predator": (0.9, 1.6, "lin"),                   # the speed of the preys is 1
    "perception_prey": (3.0, 30.0, "log"),
    "perception_predator": (5.0, 50.0, "log"),
    "distance_danger": (2.0, 15.0, "lin"),
    "energy_gain_prey": (0.3, 3.0, "log"),                 # growth rate of the preys
    "reproduction_energy_cost_prey": (20, 120, "lin"),
    "energy_loss_predator": (0.3, 2.5, "log"),             # death rate of the predators
    "reproduction_energy_needed_predator": (60, 150, "lin"),
    "reproduction_cost_fraction_predator": (0.2, 0.7, "lin"),  # cost / energy needed
    "carcass_eating_rate": (3, 30, "log"),
    "carcass_duration": (10, 60, "int"),
}

# fixed parameters
FIXED = {
    "speed_prey": 1.0,
    "max_predators": 250,   # safety limit: the predators must be limited by their food, not by this cap
    "max_energy_prey": 150,
    "max_energy_predator": 150,
    "reproduction_energy_needed_prey": 150,
}

BURN_IN = 500  # first time steps ignored by the metrics (transient)


def sample_params(rng):
    """Draw a random configuration in PARAM_SPACE."""
    params = {}
    for name, (low, high, scale) in PARAM_SPACE.items():
        if scale == "log":
            value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif scale == "int":
            value = int(rng.integers(low, high + 1))
        else:
            value = float(rng.uniform(low, high))
        params[name] = value
    return params


def perturb_params(params, rng, strength=0.15):
    """Small random variation of a configuration (multiplicative noise), kept inside PARAM_SPACE."""
    new = {}
    for name, (low, high, scale) in PARAM_SPACE.items():
        value = params[name] * np.exp(rng.normal(0, strength))
        value = min(max(value, low), high)
        new[name] = int(round(value)) if scale == "int" else float(value)
    return new


def make_world(params):
    """Create a world with the given configuration and its initial agents."""
    world = World(params["size"], params["size"])
    for name, value in FIXED.items():
        setattr(world, name, value)
    for name in (
        "carrying_capacity_prey", "speed_predator", "perception_prey", "perception_predator", "distance_danger",
        "energy_gain_prey", "reproduction_energy_cost_prey", "energy_loss_predator",
        "reproduction_energy_needed_predator", "carcass_eating_rate", "carcass_duration",
    ):
        setattr(world, name, params[name])
    world.reproduction_energy_cost_predator = (
        params["reproduction_cost_fraction_predator"] * params["reproduction_energy_needed_predator"]
    )
    world.add_prey(number=params["carrying_capacity_prey"] // 2)
    world.add_predator(number=params["n_predators_init"])
    return world


def simulate(params, seed, steps):
    """Run one simulation; return the number of preys and predators at each time step (stops at an extinction)."""
    np.random.seed(seed)
    world = make_world(params)
    preys = np.zeros(steps, dtype=int)
    predators = np.zeros(steps, dtype=int)
    for t in range(steps):
        out = world.step()
        preys[t], predators[t] = out[0], out[1]
        if out[0] == 0 or out[1] == 0:
            return preys[: t + 1], predators[: t + 1]
    return preys, predators


def autocorrelation(x):
    """Normalized autocorrelation of a time series (lags 0 to len(x) - 1)."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def dominant_period(x):
    """
    Period of the oscillations of x, from its autocorrelation: the highest peak after the first
    zero crossing, looking for at least 3 cycles in the series.
    return: period (nan if none), height of the peak (0 if none)
    """
    acf = autocorrelation(x)
    negative = np.flatnonzero(acf < 0)
    end = len(x) // 3
    if len(negative) == 0 or negative[0] >= end:
        return np.nan, 0.0
    segment = acf[negative[0]:end]
    k = int(segment.argmax())
    return float(negative[0] + k), float(max(segment[k], 0))


def predator_lag(preys, predators, max_lag):
    """Delay (in time steps) that best aligns the predator curve on the prey curve."""
    p = (preys - preys.mean()) / (preys.std() + 1e-12)
    q = (predators - predators.mean()) / (predators.std() + 1e-12)
    # a delay of one period is the same as no delay, so the delays go from 0 to max_lag - 1
    max_lag = min(int(max_lag), len(p) // 2)
    correlations = [np.mean(p[: len(p) - k] * q[k:]) for k in range(max(max_lag, 1))]
    return int(np.argmax(correlations))


def oscillation_metrics(preys, predators, steps, carrying_capacity_prey, max_predators):
    """
    Score in [0, 1] of how much the populations oscillate like a Lotka-Volterra model, and its components.
    The score is 0 if a species dies out.
    """
    extinct = "" if len(preys) == steps else ("preys" if preys[-1] == 0 else "predators")
    metrics = dict(survived=int(len(preys) == steps), extinct=extinct, survival_time=len(preys), score=0.0,
                   period=np.nan, acf_preys=0.0, acf_predators=0.0, cv_preys=0.0, cv_predators=0.0,
                   lag_fraction=np.nan, cap_preys=0.0, cap_predators=0.0,
                   mean_preys=float(np.mean(preys)), mean_predators=float(np.mean(predators)))
    if not metrics["survived"]:
        return metrics

    p = preys[BURN_IN:].astype(float)
    q = predators[BURN_IN:].astype(float)
    period_p, acf_p = dominant_period(p)
    period_q, acf_q = dominant_period(q)
    metrics.update(
        period=period_p, acf_preys=acf_p, acf_predators=acf_q,
        cv_preys=float(p.std() / p.mean()), cv_predators=float(q.std() / q.mean()),
        # fraction of the time spent near the maximum number of agents
        cap_preys=float(np.mean(p >= 0.9 * carrying_capacity_prey)),
        cap_predators=float(np.mean(q >= 0.95 * max_predators)),
        mean_preys=float(p.mean()), mean_predators=float(q.mean()),
    )
    if np.isnan(period_p) or np.isnan(period_q):
        return metrics

    lag_fraction = predator_lag(p, q, period_p) / period_p
    metrics["lag_fraction"] = lag_fraction

    # clear periodicity of both populations
    periodicity = min(acf_p, acf_q)
    # large oscillations (saturates at a coefficient of variation of 0.3)
    amplitude = min(1.0, min(metrics["cv_preys"], metrics["cv_predators"]) / 0.3)
    # the populations must not be stuck at their maximum
    no_cap = (1 - metrics["cap_preys"]) * (1 - metrics["cap_predators"])
    # Lotka-Volterra: the predators peak after the preys (a quarter of period in the classical model)
    phase = 1.0 if 0.05 <= lag_fraction <= 0.5 else 0.3
    # both populations oscillate with the same period
    same_period = 1.0 if abs(period_p - period_q) <= 0.25 * period_p else 0.5

    metrics["score"] = float(periodicity * amplitude * no_cap * phase * same_period)
    return metrics


def evaluate(task):
    """Run one (configuration, seed) and return a row of results (used by the worker processes)."""
    config_id, params, seed, steps = task
    start = time.perf_counter()
    preys, predators = simulate(params, seed, steps)
    metrics = oscillation_metrics(preys, predators, steps, params["carrying_capacity_prey"], FIXED["max_predators"])
    return dict(config_id=config_id, seed=seed, steps=steps, **params, **metrics,
                seconds=round(time.perf_counter() - start, 1))


def run_tasks(tasks, output, workers):
    """Evaluate the tasks in parallel and append each result to the CSV file as soon as it is done."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fields = ["config_id", "seed", "steps", *PARAM_SPACE, "survived", "extinct", "survival_time", "score", "period",
              "acf_preys", "acf_predators", "cv_preys", "cv_predators", "lag_fraction", "cap_preys",
              "cap_predators", "mean_preys", "mean_predators", "seconds"]
    start = time.perf_counter()
    with open(output, "w", newline="") as file, Pool(workers) as pool:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(evaluate, tasks), 1):
            writer.writerow(row)
            file.flush()
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} simulations ({time.perf_counter() - start:.0f} s)", flush=True)


def load_configs(paths):
    """Read one or several results CSV: return {config_id: (params, mean score, survival rate)}."""
    if isinstance(paths, str):
        paths = [paths]
    rows = [row for path in paths for row in csv.DictReader(open(path, newline=""))]
    configs = {}
    for row in rows:
        params = {name: (int(float(row[name])) if scale == "int" else float(row[name]))
                  for name, (_, _, scale) in PARAM_SPACE.items()}
        configs.setdefault(row["config_id"], [params, []])[1].append(row)
    result = {}
    for config_id, (params, runs) in configs.items():
        scores = [float(r["score"]) for r in runs]
        survived = [int(r["survived"]) for r in runs]
        result[config_id] = (params, float(np.mean(scores)), float(np.mean(survived)))
    return result


def ranked(configs):
    """Configurations sorted by mean score (best first)."""
    return sorted(configs.items(), key=lambda item: item[1][1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="random search in PARAM_SPACE")
    search.add_argument("--configs", type=int, default=400)
    search.add_argument("--seeds", type=int, default=2)
    search.add_argument("--steps", type=int, default=6000)
    search.add_argument("--output", default="results/search.csv")

    refine = sub.add_parser("refine", help="re-evaluate the best configurations and variations around them")
    refine.add_argument("--input", nargs="+", default=["results/search.csv"])
    refine.add_argument("--top", type=int, default=10)
    refine.add_argument("--variants", type=int, default=8, help="random variations around each configuration")
    refine.add_argument("--seeds", type=int, default=4)
    refine.add_argument("--steps", type=int, default=8000)
    refine.add_argument("--output", default="results/refine.csv")

    plot = sub.add_parser("plot", help="plot the populations of a configuration")
    plot.add_argument("--input", nargs="+", default=["results/refine.csv"])
    plot.add_argument("--rank", type=int, default=1, help="1 = best configuration of the file")
    plot.add_argument("--seeds", type=int, default=3)
    plot.add_argument("--steps", type=int, default=8000)
    plot.add_argument("--output", default="results/best.png")

    for p in (search, refine):
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
        p.add_argument("--rng-seed", type=int, default=0, help="seed used to draw the configurations")

    args = parser.parse_args()

    if args.command == "search":
        rng = np.random.default_rng(args.rng_seed)
        tasks = []
        for i in range(args.configs):
            params = sample_params(rng)
            # the id contains the rng seed, so that the ids of several searches don't collide
            tasks += [(f"r{args.rng_seed}-{i}", params, seed, args.steps) for seed in range(args.seeds)]
        run_tasks(tasks, args.output, args.workers)

    elif args.command == "refine":
        rng = np.random.default_rng(args.rng_seed)
        tasks = []
        for config_id, (params, _, _) in ranked(load_configs(args.input))[: args.top]:
            variants = [params] + [perturb_params(params, rng) for _ in range(args.variants)]
            for j, variant in enumerate(variants):
                # seeds different from the search, to check that the result is not a lucky draw
                tasks += [(f"{config_id}.{j}", variant, 1000 + seed, args.steps) for seed in range(args.seeds)]
        run_tasks(tasks, args.output, args.workers)

    elif args.command == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config_id, (params, score, survival) = ranked(load_configs(args.input))[args.rank - 1]
        fig, axes = plt.subplots(args.seeds, 2, figsize=(14, 3.2 * args.seeds), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[3, 1]))
        for seed, (ax, ax_phase) in zip(range(args.seeds), axes):
            preys, predators = simulate(params, 2000 + seed, args.steps)
            ax.plot(preys, color="blue", label="Preys")
            ax.set_ylabel("Preys", color="blue")
            ax2 = ax.twinx()
            ax2.plot(predators, color="orange")
            ax2.set_ylabel("Predators", color="orange")
            ax.axvline(BURN_IN, color="gray", ls=":", lw=1)
            ax.set_title(f"seed {2000 + seed}")
            ax_phase.plot(preys[BURN_IN:], predators[BURN_IN:], lw=0.6, color="purple")
            ax_phase.set_xlabel("Preys")
            ax_phase.set_ylabel("Predators")
            ax_phase.set_title("Phase portrait")
        axes[-1][0].set_xlabel("Time")
        fig.suptitle(f"config {config_id} (mean score {score:.2f}, survival {survival:.0%})")
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fig.savefig(args.output, dpi=90)
        with open(os.path.splitext(args.output)[0] + ".json", "w") as file:
            json.dump(dict(config_id=config_id, mean_score=score, survival=survival, params=params, fixed=FIXED),
                      file, indent=2)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
