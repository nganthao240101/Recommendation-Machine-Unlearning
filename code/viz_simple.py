import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

r = {
 "Random": {"r10": 0.11359, "r20": 0.18195, "r50": 0.31794, "p10": 0.25421, "p20": 0.21459, "p50": 0.16168, "n10": 0.28371, "n20": 0.27471, "n50": 0.29634, "c": "#888888"},
 "InBP": {"r10": 0.12372, "r20": 0.20243, "r50": 0.35447, "p10": 0.26682, "p20": 0.22949, "p50": 0.17440, "n10": 0.29601, "n20": 0.29232, "n50": 0.32083, "c": "#A23B72"},
 "UBP": {"r10": 0.13541, "r20": 0.21844, "r50": 0.37638, "p10": 0.28240, "p20": 0.24331, "p50": 0.18402, "n10": 0.31306, "n20": 0.31114, "n50": 0.34155, "c": "#2E86AB"}
}
m = ["Random", "InBP", "UBP"]

fig, ax = plt.subplots(figsize=(10, 6))
vals = [r[x]["r20"] for x in m]
cols = [r[x]["c"] for x in m]
bars = ax.bar(m, vals, color=cols, edgecolor="black", linewidth=1.5)
for b, v in zip(bars, vals):
 ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Recall@20", fontsize=12)
ax.set_title("RecEraser BPR on ml-1m: Recall@20 by Partition Method", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 0.25)
plt.tight_layout()
plt.savefig("../results/recall20_comparison.png", dpi=120, bbox_inches="tight")
print("OK saved")
plt.close()
