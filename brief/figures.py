import json

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = str(HERE) + "/"
REPO = str(HERE.parent) + "/"
INK = "#1F2937"
MUTED = "#6B7280"
BLUE = "#DBEAFE"
BLUE_E = "#1D4ED8"
GREEN = "#DCFCE7"
GREEN_E = "#15803D"
AMB = "#FEF3C7"
AMB_E = "#B45309"
GREY = "#F3F4F6"
GREY_E = "#6B7280"
RED = "#FEE2E2"
RED_E = "#B91C1C"
VIO = "#EDE9FE"
VIO_E = "#6D28D9"


def box(ax, x, y, w, h, title, body, fc, ec, fs=7.6, tag=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fc, ec=ec, lw=1.1
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.022,
        title,
        ha="center",
        va="top",
        fontsize=fs + 1.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        x + w / 2,
        y + h - 0.068,
        body,
        ha="center",
        va="top",
        fontsize=fs,
        color=INK,
        linespacing=1.32,
    )
    if tag:
        ax.text(
            x + w - 0.006,
            y + 0.012,
            tag,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color=ec,
            fontweight="bold",
        )


def arr(ax, a, b, label="", color=INK, rad=0.0, ls="-", lx=0, ly=0, fs=6.8):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.0,
            color=color,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (a[0] + b[0]) / 2 + lx,
            (a[1] + b[1]) / 2 + ly,
            label,
            fontsize=fs,
            color=color,
            ha="center",
            va="center",
            bbox=dict(fc="white", ec="none", pad=0.5),
        )


# ---------------- Figure 1: integrated architecture ----------------
fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.add_patch(
    FancyBboxPatch(
        (0.004, 0.70),
        0.992,
        0.292,
        boxstyle="round,pad=0.003",
        fc="none",
        ec=BLUE_E,
        ls="--",
        lw=0.8,
    )
)
ax.text(
    0.012,
    0.985,
    "DURING A RUN  ·  critical path  ·  0 LLM calls by memory",
    fontsize=7.8,
    color=BLUE_E,
    fontweight="bold",
    va="top",
)
ax.add_patch(
    FancyBboxPatch(
        (0.004, 0.005),
        0.992,
        0.488,
        boxstyle="round,pad=0.003",
        fc="none",
        ec=GREEN_E,
        ls="--",
        lw=0.8,
    )
)
ax.text(
    0.99,
    0.481,
    "AFTER THE RUN, ON FEEDBACK, NIGHTLY  ·  off the critical path",
    fontsize=7.8,
    color=GREEN_E,
    fontweight="bold",
    va="top",
    ha="right",
)

# critical path
box(ax, 0.015, 0.715, 0.13, 0.235, "Task", "one tenant\nper task", GREY, GREY_E)
box(
    ax,
    0.185,
    0.715,
    0.20,
    0.235,
    "Recall",
    "labels each line: ACTIVE,\nCONTESTED, STALE-REVALIDATE,\nCANDIDATE (do not act),\nANTI-MEMORY · ≤3,600 tokens\nlogs what it showed (exposure)",
    BLUE,
    BLUE_E,
    fs=7.0,
)  # ,BLUE,BLUE_E)
box(ax, 0.425, 0.715, 0.15, 0.235, "Agent", "black-box LLM\n+ tools", GREY, GREY_E)
box(
    ax,
    0.615,
    0.715,
    0.23,
    0.235,
    "ACTION GATE",
    "before every side effect:\nrisk class · predicate authority\n· freshness · target allowlist\n· anti-memory veto\n· idempotency key",
    RED,
    RED_E,
)
box(ax, 0.875, 0.715, 0.11, 0.235, "Effect", "email,\nconfig,\nticket", GREY, GREY_E)
arr(ax, (0.145, 0.83), (0.185, 0.83))
arr(ax, (0.385, 0.83), (0.425, 0.83), "context")
arr(ax, (0.575, 0.83), (0.615, 0.83), "call")
arr(ax, (0.845, 0.83), (0.875, 0.83), "allow")

# evidence inputs (middle band)
box(ax, 0.185, 0.505, 0.80, 0.17, "", "", VIO, VIO_E)
ax.text(
    0.20,
    0.66,
    "Three separate properties of every piece of evidence",
    fontsize=8.2,
    fontweight="bold",
    color=VIO_E,
    va="top",
)
ax.text(
    0.20,
    0.615,
    "① Authenticity: HMAC seal binds connector, tenant, run and content digest.",
    fontsize=7.4,
    color=INK,
    va="top",
)
ax.text(
    0.20,
    0.578,
    "② Predicate authority: CRM verifies contacts, infra API versions/config; else capped.",
    fontsize=7.4,
    color=INK,
    va="top",
)
ax.text(
    0.20,
    0.541,
    "③ Freshness: last confirmed vs window (routing 30 d, operational 90 d, diagnostic 180 d).",
    fontsize=7.4,
    color=INK,
    va="top",
)
arr(ax, (0.73, 0.675), (0.73, 0.715), color=VIO_E)
arr(ax, (0.33, 0.505), (0.33, 0.447), color=VIO_E)

# off critical path
box(
    ax,
    0.015,
    0.235,
    0.18,
    0.212,
    "Write path  [≤1 LLM]",
    "structured output → claims\nprose → atomic claims,\neach bound to its quote\n(+1 retry if partly covered)",
    AMB,
    AMB_E,
)
box(
    ax,
    0.225,
    0.235,
    0.20,
    0.212,
    "BELIEF GATE  [0 LLM]",
    "admission caps by trust\none ACTIVE value per slot\nconflict → CONTESTED\n(both observations kept)",
    AMB,
    AMB_E,
)
box(
    ax,
    0.455,
    0.235,
    0.22,
    0.212,
    "Claim store (SQLite WAL)",
    "run_reads = EXPOSURE log\nderived_from = CAUSAL DAG\n(declared parents only)\n+ refutation ledger",
    BLUE,
    BLUE_E,
)
box(
    ax,
    0.705,
    0.235,
    0.28,
    0.212,
    "Nightly  [≤1 LLM]",
    "global: ≥5 tenants, ≥3 independent\nsources, same value; polarity and\nversions checked against evidence\ncompaction · TTL · 500-claim cap",
    GREEN,
    GREEN_E,
)
arr(ax, (0.195, 0.33), (0.225, 0.33))
arr(ax, (0.425, 0.33), (0.455, 0.33))
arr(ax, (0.675, 0.33), (0.705, 0.33))
arr(ax, (0.08, 0.70), (0.08, 0.455), "observations\n+ exposure\nlog", color=MUTED, fs=6.6)

# feedback path
box(ax, 0.015, 0.02, 0.97, 0.185, "", "", RED, RED_E)
ax.text(
    0.03,
    0.198,
    "Delayed human rejection (~2% of runs, days later)",
    fontsize=8.2,
    fontweight="bold",
    color=RED_E,
    va="top",
)
steps = [
    "attribution over the\nrun's exposed + created\nclaims  [1 LLM]",
    "root named by the\nfeedback → REFUTED\n+ anti-memory",
    "unsupported roots →\nCONTESTED, not\nrefuted",
    "declared descendants\n→ QUARANTINED",
    "independently verified\ndescendants → kept\nACTIVE",
    "affected runs\nflagged for\nreview",
]
xs = [0.03, 0.19, 0.35, 0.51, 0.665, 0.83]
for i, (x, t) in enumerate(zip(xs, steps, strict=False)):
    ax.text(x + 0.07, 0.145, t, fontsize=7.0, color=INK, ha="center", va="top", linespacing=1.3)
    if i < len(xs) - 1:
        arr(ax, (x + 0.142, 0.115), (xs[i + 1] - 0.002, 0.115), color=RED_E)
arr(ax, (0.565, 0.235), (0.565, 0.208), color=RED_E)
fig.savefig(OUT + "fig1_architecture.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Figure 2: canonical measurements ----------------
b = json.loads(Path(REPO + "benchmarks/results.json").read_text())
t = json.loads(Path(REPO + "eval/results/three_arm_offline.json").read_text())
S1, S2, S3 = "#257de0", "#f47230", "#00b894"
TXT = "#191919"
TXT2 = "#666666"
GRID = "#e6e6e6"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": TXT2,
        "axes.labelcolor": TXT2,
        "xtick.color": TXT2,
        "ytick.color": TXT2,
    }
)
fig, axs = plt.subplots(
    1, 3, figsize=(10.5, 2.9), dpi=220, gridspec_kw={"width_ratios": [1.05, 0.85, 1.35]}
)
for a in axs:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)

g = b["two_year_growth_trajectory"]
days = [x["day"] for x in g]
a = axs[0]
a.plot(
    days, [x["total_memories"] for x in g], color=S1, lw=2, marker="o", ms=4, label="stored claims"
)
a.plot(days, [x["active"] for x in g], color=S2, lw=2, marker="o", ms=4, label="active claims")
a.set_xticks(days)
a.set_xlabel("simulated day")
a.set_ylabel("claims")
a.legend(frameon=False, fontsize=7, loc="center right")
a.set_title(
    "(a) Retention, 730 simulated days\n20 tenants, 17,544 writes",
    fontsize=8.2,
    color=TXT,
    loc="left",
)
a.text(
    730,
    g[-1]["total_memories"] + 250,
    f"{g[-1]['disk_mb']:.1f} MB",
    fontsize=7,
    color=TXT2,
    ha="right",
)

s = b["scale_and_latency"]
a = axs[1]
labels = ["cold\nconnection", "warm\nconnection"]
p50 = [
    s["cold_connection_pass"]["cumulative_run_critical_path_ms"]["p50"],
    s["warm_pass"]["cumulative_run_critical_path_ms"]["p50"],
]
p95 = [
    s["cold_connection_pass"]["cumulative_run_critical_path_ms"]["p95"],
    s["warm_pass"]["cumulative_run_critical_path_ms"]["p95"],
]
xs = [0, 1]
w = 0.34
a.bar([x - w / 2 - 0.01 for x in xs], p50, w, color=S1, label="p50")
a.bar([x + w / 2 + 0.01 for x in xs], p95, w, color=S2, label="p95")
for x, v in zip(xs, p95, strict=False):
    a.text(x + w / 2 + 0.01, v + 6, f"{v:.0f}", ha="center", fontsize=7, color=TXT)
a.axhline(300, color=TXT2, ls="--", lw=1)
a.text(1.45, 300, "300 ms budget", fontsize=6.8, color=TXT2, ha="right", va="bottom")
a.set_xticks(xs, labels)
a.set_ylim(0, 330)
a.set_ylabel("ms per run (all steps)")
a.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.62))
a.set_title(
    "(b) Memory critical path\n62k claims, 8 workers, 1 host", fontsize=8.2, color=TXT, loc="left"
)

arms = [
    ("memory", "Memory"),
    ("stateless_holdout", "Stateless"),
    ("naive_rag", "RAG +\ngate"),
    ("naive_rag_ungated", "RAG, no\ngate"),
    ("memory_no_revalidation", "Memory, no\nstale labels"),
]
A = t["arms"]
a = axs[2]
corr = [100 * A[k]["correct_final_answer"]["value"] for k, _ in arms]
fmu = [100 * (A[k]["false_memory_utilization"]["value"] or 0) for k, _ in arms]
xs = list(range(len(arms)))
a.bar([x - w / 2 - 0.01 for x in xs], corr, w, color=S1, label="correct final answer")
a.bar(
    [x + w / 2 + 0.01 for x in xs],
    fmu,
    w,
    color=S2,
    label="false memory used (% of runs shown one)",
)
for x, v in zip(xs, corr, strict=False):
    a.text(x - w / 2 - 0.01, v + 2, f"{v:.0f}", ha="center", fontsize=6.8, color=TXT)
for x, (k, _), v in zip(xs, arms, fmu, strict=False):
    lab = "n/a" if k == "stateless_holdout" else f"{v:.0f}"
    a.text(x + w / 2 + 0.01, v + 2, lab, ha="center", fontsize=6.8, color=TXT)
a.set_xticks(xs, [lbl for _, lbl in arms], fontsize=6.9)
a.set_ylim(0, 140)
a.set_yticks([0, 20, 40, 60, 80, 100])
a.set_ylabel("% of runs")
a.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
a.set_title("(c) Offline pilot, simulated agent, 15 tasks/arm", fontsize=8.2, color=TXT, loc="left")
fig.tight_layout(w_pad=1.6)
fig.savefig(OUT + "fig2_measurements.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("ok")
