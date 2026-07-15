"""End-to-end dry run of the full protocol on the mock ecosystem.

Runs gate-free calibration and test episodes, fits the gate, reports
lambda-hat, realized selective risk and coverage, empirical validity, mean
cost, and the single-pass baseline for contrast. No model weights needed.

Usage:  python scripts/demo_mock.py [--regions 600] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ugra.agent.loop import LoopConfig, run_episode
from ugra.agent.planner import RulePlanner
from ugra.calibration.gate import calibrate, price_trajectory
from ugra.metrics.edit import cer
from ugra.metrics.selective import mean_cost, realized_at, validity_hat
from ugra.mock import make_regions, make_toolbox
from ugra.tools.base import ReadContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--budget", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-accept", type=int, default=25,
                        help="preregistered scan-start floor (Section 3.4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("demo")

    toolbox = make_toolbox()
    costs = {t.name: t.cost for t in (toolbox.specialist, toolbox.generalist,
                                      *toolbox.restorers, *toolbox.verifiers)}
    planner = RulePlanner(costs)
    config = LoopConfig(budget=args.budget)
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 2)

    regions = make_regions(args.regions, seed=args.seed)
    half = len(regions) // 2
    log.info("running %d gate-free episodes (rule planner, B=%.0f) ...",
             len(regions), args.budget)
    trajectories = [run_episode(r, toolbox, planner, config, root_quality=q)[0]
                    for r, q in regions]
    priced = [price_trajectory(t, grid) for t in trajectories]
    costs = [t.total_cost for t in trajectories]

    gate = calibrate(priced[:half], args.alpha, args.delta, grid,
                     min_accept=args.min_accept)
    log.info("calibrated threshold lambda-hat = %s", gate.lambda_hat)

    risk, coverage = realized_at(priced[half:], gate.lambda_hat, grid)
    v_hat = validity_hat(priced, args.alpha, args.delta, grid,
                         n_resamples=50, seed=args.seed)
    agent_cost = mean_cost(priced[half:], gate.lambda_hat, grid, costs[half:])

    single_pass_losses = []
    for region, quality in regions[half:]:
        text, _ = toolbox.specialist.read(None, ReadContext(
            region_id=region.region_id, view_id=0, quality=quality,
            reference=region.reference))
        single_pass_losses.append(cer(text, region.reference))

    log.info("---- test split ----")
    log.info("realized selective risk  : %.4f  (target alpha = %.2f)", risk, args.alpha)
    log.info("certified coverage       : %.1f%%", 100 * coverage)
    log.info("empirical validity V-hat : %.2f  (requires >= %.2f)", v_hat, 1 - args.delta)
    log.info("mean cost per region     : %.2f units", agent_cost)
    log.info("single-pass specialist CER (all regions): %.4f",
             float(np.mean(single_pass_losses)))


if __name__ == "__main__":
    main()
