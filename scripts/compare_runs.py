#!/usr/bin/env python
"""Aggregate eval JSONLs and emit a comparison table (mean reward, win-rate vs base)."""
import argparse, json
from pathlib import Path
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_files", nargs="+", help="One or more eval JSONLs (each row {name, id, scores}).")
    ap.add_argument("--baseline", default=None, help="Name of the baseline run (default: first file's name).")
    ap.add_argument("--out", default="runs/eval/summary.json")
    args = ap.parse_args()

    by_name = defaultdict(list)
    for f in args.eval_files:
        for row in load(f):
            by_name[row["name"]].append(row)

    if not by_name:
        print("no rows")
        return

    base_name = args.baseline or list(by_name.keys())[0]
    if base_name not in by_name:
        raise SystemExit(f"baseline {base_name} not in {list(by_name.keys())}")

    base_by_id = {r["id"]: r for r in by_name[base_name]}

    summary = {}
    DIMS = ["VQ", "MQ", "TA", "Overall"]
    print(f"{'run':<24} | " + " | ".join(f"{d:>9} (Δ vs {base_name[:6]})" for d in DIMS) + " |  win%")
    print("-" * 120)
    for name, rows in by_name.items():
        means = {k: sum(r["scores"][k] for r in rows) / len(rows) for k in DIMS}
        deltas = {k: means[k] - sum(r["scores"][k] for r in by_name[base_name]) / len(by_name[base_name]) for k in DIMS}
        wins = {}
        for k in DIMS:
            cnt, tot = 0, 0
            for r in rows:
                if r["id"] not in base_by_id: continue
                tot += 1
                cnt += 1 if r["scores"][k] > base_by_id[r["id"]]["scores"][k] else 0
            wins[k] = cnt / max(1, tot)
        cells = " | ".join(f"{means[k]:+8.3f} ({deltas[k]:+5.3f})" for k in DIMS)
        wins_overall = wins["Overall"] * 100
        print(f"{name:<24} | {cells} | {wins_overall:5.1f}%")
        summary[name] = {"means": means, "deltas_vs_base": deltas, "winrate_vs_base": wins, "n": len(rows)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"baseline": base_name, "by_run": summary}, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
