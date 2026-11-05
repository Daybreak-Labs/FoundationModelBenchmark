#!/usr/bin/env python3
"""Pull the zero-shot FM roster from the official benchmark-scoring gate leaderboards and write anonymized scorecards.

Read-only: every number comes from benchmark_scoring.<client>.leaderboard (the gate's own view), never recomputed here.
Client names live only in .clients.local.json (git-ignored): {"Customer A": {"schema": "<scoring schema>", "freq": "Weekly", ...}, ...} (template: .clients.local.json.example)

  python3 scripts/pull_boards.py            # writes data/scorecards/*.csv, data/cross_customer.csv
Needs the databricks CLI logged in (profile via DATABRICKS_CONFIG_PROFILE) and SQL warehouse access.
"""
import csv
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = os.environ.get("FMB_WAREHOUSE_ID", "4c38d6a537acd61a")

# Display name -> gate model names, first one present on a board wins (some boards carry a _V2 re-key of the same model).
ROSTER = {
    "TimesFM 2.5": ["TIMESFM25_SL2", "TIMESFM25_SL2_V2"],
    "Chronos-2": ["CHRONOS2_SL2", "CHRONOS2_SL2_V2"],
    "Chronos-2 XL": ["CHRONOS2_XL"],
    "TiRex 2": ["TIREX2"],
    "Toto 2": ["TOTO2"],
    "Moirai 2 small": ["MOIRAI2_SMALL_MEAN_V2"],
    "Moirai-MoE base": ["MOIRAI_MOE_BASE_MEAN_V2"],
    "Moirai-MoE small": ["MOIRAI_MOE_SMALL_MEAN_V2"],
    "TTM r2": ["TTM"],
}
BASELINES = {"Seasonal Naive": ["SEASONAL_NAIVE", "SNAIVE52_BASE"]}
FIELDS = ["customer", "kind", "model", "board_rank", "board_size", "board_pct", "rank_value", "wmape", "abs_bias", "net_bias", "rmse", "fill_share"]


def sql(q: str) -> list[dict]:
    body = json.dumps({"warehouse_id": WAREHOUSE, "statement": q, "wait_timeout": "50s"})
    r = json.loads(subprocess.run(["databricks", "api", "post", "/api/2.0/sql/statements", "--json", body],
                                  check=True, capture_output=True, text=True).stdout)
    if r["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(f"{q[:80]}: {r['status']}")
    cols = [c["name"] for c in r["manifest"]["schema"]["columns"]]
    return [dict(zip(cols, row)) for row in r.get("result", {}).get("data_array", [])]


def board(customer: str, schema: str) -> list[dict]:
    rows = sql(f"SELECT rank, model, rank_value, wmape, bias, net_bias, rmse, fill_share FROM benchmark_scoring.{schema}.leaderboard")
    by_model = {r["model"]: r for r in rows}
    size = len(rows)
    out = []

    def add(kind, name, r):
        rank = int(r["rank"])
        out.append({"customer": customer, "kind": kind, "model": name, "board_rank": rank, "board_size": size,
                    "board_pct": round(rank / size, 4), "rank_value": float(r["rank_value"]), "wmape": float(r["wmape"]),
                    "abs_bias": float(r["bias"]), "net_bias": float(r["net_bias"]), "rmse": float(r["rmse"]), "fill_share": float(r["fill_share"] or 0)})

    for kind, group in (("fm", ROSTER), ("baseline", BASELINES)):
        for name, candidates in group.items():
            hit = next((by_model[m] for m in candidates if m in by_model), None)
            if hit:
                add(kind, name, hit)
    add("reference", "Board #1 (best tuned model)", min(rows, key=lambda r: int(r["rank"])))
    return out


def main():
    clients = json.loads((ROOT / ".clients.local.json").read_text())
    (ROOT / "data" / "scorecards").mkdir(parents=True, exist_ok=True)
    allrows = []
    for customer, meta in clients.items():
        rows = board(customer, meta["schema"])
        allrows += rows
        slug = customer.lower().replace(" ", "_")
        with open(ROOT / "data" / "scorecards" / f"{slug}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["board_rank"]))
        print(f"{customer}: {sum(r['kind'] == 'fm' for r in rows)}/{len(ROSTER)} FMs on a board of {rows[0]['board_size']}")

    # 'bias' in the leaderboard view is the scorer's mean |bias| (per snapshot); net_bias is the signed one.
    # Cross-customer: mean board percentile (vs everything on the board) and mean rank within the FM roster + baselines.
    named = [r for r in allrows if r["kind"] in ("fm", "baseline")]
    for c in clients:
        cr = sorted((r for r in named if r["customer"] == c), key=lambda r: r["board_rank"])
        for i, r in enumerate(cr, 1):
            r["roster_rank"] = i
    summary = []
    for name in list(ROSTER) + list(BASELINES):
        rs = [r for r in named if r["model"] == name]
        if rs:
            summary.append({"model": name, "boards": len(rs),
                            "mean_board_pct": round(sum(r["board_pct"] for r in rs) / len(rs), 3),
                            "mean_roster_rank": round(sum(r["roster_rank"] for r in rs) / len(rs), 2),
                            "mean_abs_bias": round(sum(r["abs_bias"] for r in rs) / len(rs), 3),
                            "mean_net_bias": round(sum(r["net_bias"] for r in rs) / len(rs), 3),
                            "boards_under": sum(r["net_bias"] < 0 for r in rs),
                            "max_fill_share": round(max(r["fill_share"] for r in rs), 4)})
    summary.sort(key=lambda s: s["mean_board_pct"])
    with open(ROOT / "data" / "cross_customer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
