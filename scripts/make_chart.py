#!/usr/bin/env python3
"""Render data/cross_customer_chart.png from data/cross_customer.csv (run pull_boards.py first)."""
import csv
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
FM, BASE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2; baseline also labeled "(baseline)"

rows = list(csv.DictReader(open(ROOT / "data" / "cross_customer.csv")))
n_boards = max(int(r["boards"]) for r in rows)
rows.sort(key=lambda r: float(r["mean_board_pct"]), reverse=True)  # plotly draws bottom-up: best ends on top
beaten = [1 - float(r["mean_board_pct"]) for r in rows]
names = [r["model"] + (" (baseline)" if r["model"] == "Seasonal Naive" else "")
         + ("" if int(r["boards"]) == n_boards else f"  [{r['boards']}/{n_boards} boards]") for r in rows]

fig = go.Figure(go.Bar(
    x=beaten, y=names, orientation="h", marker_color=[BASE if r["model"] == "Seasonal Naive" else FM for r in rows],
    marker_cornerradius=4, text=[f"{b:.0%}" for b in beaten], textposition="outside", textfont=dict(color=INK, size=13),
    cliponaxis=False,
))
fig.update_layout(
    title=dict(text="Share of each client board a zero-shot FM beats, averaged over customers",
               font=dict(color=INK, size=17), x=0.01),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family="Helvetica, Arial, sans-serif", color=INK2, size=13),
    width=1000, height=90 + 42 * len(rows), margin=dict(l=10, r=60, t=60, b=50), bargap=0.35, showlegend=False,
    xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=GRID, zeroline=False,
               title=dict(text="Higher is better. Board = every model on that client's official leaderboard.", font=dict(size=12))),
    yaxis=dict(automargin=True, tickfont=dict(color=INK, size=13)),
)
fig.write_image(ROOT / "data" / "cross_customer_chart.png", scale=2)
print("wrote data/cross_customer_chart.png")
