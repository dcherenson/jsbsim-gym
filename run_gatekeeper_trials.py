import argparse
import csv
import json
import shlex
import subprocess
import time
import os
from pathlib import Path
from statistics import mean, median


REPO_ROOT = Path(__file__).resolve().parent


def _float_or_none(value):
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _build_trial_command(base_command, seed, summary_path, record_video):
    cmd = shlex.split(base_command)
    cmd.extend([
        "--seed",
        str(int(seed)),
        "--record-video" if bool(record_video) else "--no-record-video",
        "--run-summary-json",
        str(summary_path),
    ])
    return cmd


def run_trials(base_command, trials, seed_start, seed_step, workdir, output_dir, record_video):
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_summary_dir = output_dir / "trial_summaries"
    trial_summary_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    rows = []
    for trial_idx in range(int(trials)):
        seed = int(seed_start + trial_idx * seed_step)
        summary_path = trial_summary_dir / f"trial_{trial_idx:03d}_summary.json"
        cmd = _build_trial_command(base_command, seed, summary_path, record_video=record_video)

        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            check=False,
            env=dict(os.environ, PYTHONHASHSEED=str(seed)),
        )
        elapsed_s = time.time() - t0

        summary = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        termination_reason = str(summary.get("termination_reason", "missing_summary"))
        end_step = int(summary.get("end_step", -1))
        speed_at_end_fps = _float_or_none(summary.get("speed_at_end_fps"))
        speed_at_end_kts = _float_or_none(summary.get("speed_at_end_kts"))
        average_speed_fps = _float_or_none(summary.get("average_speed_fps"))
        average_speed_kts = _float_or_none(summary.get("average_speed_kts"))
        average_altitude_ft = _float_or_none(summary.get("average_altitude_ft"))
        average_altitude_msl_ft = _float_or_none(summary.get("average_altitude_msl_ft"))
        percent_below_canyon_top_altitude = _float_or_none(summary.get("percent_below_canyon_top_altitude"))
        backup_steps_used = int(summary.get("backup_steps_used", 0))
        nominal_progress_fraction = _float_or_none(summary.get("nominal_progress_fraction"))
        mission_success = bool(summary.get("mission_success", False))

        if not mission_success:
            failures += 1

        row = {
            "trial_idx": trial_idx,
            "seed": seed,
            "return_code": int(result.returncode),
            "elapsed_s": float(elapsed_s),
            "termination_reason": termination_reason,
            "failure_step": end_step,
            "speed_at_failure_fps": speed_at_end_fps,
            "speed_at_failure_kts": speed_at_end_kts,
            "average_speed_fps": average_speed_fps,
            "average_speed_kts": average_speed_kts,
            "average_altitude_ft": average_altitude_ft,
            "average_altitude_msl_ft": average_altitude_msl_ft,
            "percent_below_canyon_top_altitude": percent_below_canyon_top_altitude,
            "backup_steps_used": backup_steps_used,
            "nominal_progress_fraction": nominal_progress_fraction,
            "mission_success": mission_success,
            "summary_json": str(summary_path),
            "stderr_tail": "\n".join(result.stderr.splitlines()[-8:]) if result.stderr else "",
        }
        rows.append(row)

        print(
            f"[{trial_idx + 1:03d}/{trials:03d}] total failures={failures} seed={seed} rc={result.returncode} "
            f"step={end_step} speed_kts={speed_at_end_kts if speed_at_end_kts is not None else 'nan'} "
            f"avg_speed_kts={average_speed_kts if average_speed_kts is not None else 'nan'} "
            f"avg_alt_ft={average_altitude_ft if average_altitude_ft is not None else 'nan'} "
            f"below_top_pct={percent_below_canyon_top_altitude if percent_below_canyon_top_altitude is not None else 'nan'} "
            f"prog={nominal_progress_fraction if nominal_progress_fraction is not None else 'nan'} "
            f"backup_steps={backup_steps_used} success={mission_success} reason={termination_reason}"
        )

    return rows


def _compute_aggregate(rows):
    rc_ok = [r for r in rows if int(r["return_code"]) == 0]
    mission_success_rows = [r for r in rc_ok if bool(r.get("mission_success", False))]
    failure_steps = [int(r["failure_step"]) for r in rc_ok if int(r["failure_step"]) >= 0]
    backup_steps = [int(r["backup_steps_used"]) for r in rc_ok]
    speeds_kts = [float(r["speed_at_failure_kts"]) for r in rc_ok if r["speed_at_failure_kts"] is not None]
    average_speeds_kts = [float(r["average_speed_kts"]) for r in rc_ok if r["average_speed_kts"] is not None]
    average_altitudes_ft = [float(r["average_altitude_ft"]) for r in rc_ok if r["average_altitude_ft"] is not None]
    below_top_percentages = [
        float(r["percent_below_canyon_top_altitude"])
        for r in rc_ok
        if r["percent_below_canyon_top_altitude"] is not None
    ]
    progress_fractions = [float(r["nominal_progress_fraction"]) for r in rc_ok if r["nominal_progress_fraction"] is not None]

    return {
        "num_trials": int(len(rows)),
        "num_successful_commands": int(len(rc_ok)),
        "num_failed_commands": int(len(rows) - len(rc_ok)),
        "num_mission_successes": int(len(mission_success_rows)),
        "failure_step_mean": mean(failure_steps) if failure_steps else None,
        "failure_step_median": median(failure_steps) if failure_steps else None,
        "backup_steps_mean": mean(backup_steps) if backup_steps else None,
        "backup_steps_median": median(backup_steps) if backup_steps else None,
        "speed_at_failure_kts_mean": mean(speeds_kts) if speeds_kts else None,
        "speed_at_failure_kts_median": median(speeds_kts) if speeds_kts else None,
        "average_speed_kts_mean": mean(average_speeds_kts) if average_speeds_kts else None,
        "average_speed_kts_median": median(average_speeds_kts) if average_speeds_kts else None,
        "average_altitude_ft_mean": mean(average_altitudes_ft) if average_altitudes_ft else None,
        "average_altitude_ft_median": median(average_altitudes_ft) if average_altitudes_ft else None,
        "percent_below_canyon_top_altitude_mean": mean(below_top_percentages) if below_top_percentages else None,
        "percent_below_canyon_top_altitude_median": median(below_top_percentages) if below_top_percentages else None,
        "nominal_progress_fraction_mean": mean(progress_fractions) if progress_fractions else None,
        "nominal_progress_fraction_median": median(progress_fractions) if progress_fractions else None,
    }


def write_outputs(rows, output_csv, output_json, base_command):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial_idx",
        "seed",
        "return_code",
        "elapsed_s",
        "termination_reason",
        "failure_step",
        "speed_at_failure_fps",
        "speed_at_failure_kts",
        "average_speed_fps",
        "average_speed_kts",
        "average_altitude_ft",
        "average_altitude_msl_ft",
        "percent_below_canyon_top_altitude",
        "backup_steps_used",
        "nominal_progress_fraction",
        "mission_success",
        "summary_json",
        "stderr_tail",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = _compute_aggregate(rows)
    payload = {
        "command": base_command,
        "aggregate": aggregate,
        "rows": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return aggregate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a gatekeeper CLI command repeatedly with varying seeds and collect "
            "failure step, speed, altitude, and backup-usage metrics."
        )
    )
    parser.add_argument(
        "--command",
        # required=True,
        help=(
            "Base command for a single run, e.g. \"uv run python run_scenario.py "
            "--controller pid_traj --gatekeeper --nominal-dyn-path ...\"."
        ),
        default="uv run python run_scenario.py --gatekeeper --nominal-dyn-path air-racing-optimization/final_results/f16dyn_crude.asb --nominal-end-fraction 0.1 --controller pid_traj"
    )
    parser.add_argument("--trials", type=int, default=100, help="Number of repeated runs.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value.")
    parser.add_argument("--seed-step", type=int, default=1, help="Step between trial seeds.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=REPO_ROOT,
        help="Working directory for command execution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "gatekeeper_trials",
        help="Directory where trial summaries and aggregate outputs are written.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit JSON output path.",
    )
    parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable video recording for each trial command.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv) if args.output_csv is not None else output_dir / "gatekeeper_trials.csv"
    output_json = Path(args.output_json) if args.output_json is not None else output_dir / "gatekeeper_trials.json"

    rows = run_trials(
        base_command=str(args.command),
        trials=int(args.trials),
        seed_start=int(args.seed_start),
        seed_step=int(args.seed_step),
        workdir=Path(args.workdir),
        output_dir=output_dir,
        record_video=bool(args.record_video),
    )
    aggregate = write_outputs(rows, output_csv=output_csv, output_json=output_json, base_command=str(args.command))

    print("\nAggregate summary:")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print(f"\nSaved trial table: {output_csv}")
    print(f"Saved aggregate JSON: {output_json}")


if __name__ == "__main__":
    main()
