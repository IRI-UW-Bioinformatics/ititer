#!/usr/bin/env python3

# %%

import matplotlib.pyplot as plt
import ititer as it


df = it.load_example_data()
df.head(10).round(3).to_csv("long-example.csv")

wide = df.head(24).pivot(columns="Sample", values="OD", index="Dilution").T
wide.columns.name = None
wide.round(3).to_csv("wide-example.csv")

df["Log Dilution"] = it.titer_to_index(df["Dilution"], start=40, fold=4)

sigmoid = it.Sigmoid(a="partial", b="full", c=0, d="full")

sigmoid = sigmoid.fit(
    log_dilution=df["Log Dilution"],
    response=df["OD"],
    sample_labels=df["Sample"],
)

sigmoid.plot_sample("21-P0004-v001sr01", step=1000)
plt.savefig("1-sample.png", bbox_inches="tight")
plt.close()

sigmoid.plot_sample("21-P0004-v001sr01", mean=True)
plt.savefig("1-sample-mean.png", bbox_inches="tight")
plt.close()

sigmoid.plot_samples(["21-P0833-v001sr01", "21-P0834-v001sr01"])
plt.savefig("2-samples.png", bbox_inches="tight")
plt.close()

sigmoid.plot_all_samples()
plt.savefig("all-samples.png", bbox_inches="tight")
plt.close()

df_inflections = sigmoid.inflections(hdi_prob=0.95)
df_inflections.head().round(2).to_csv("inflections-example.csv")

it.index_to_titer(df_inflections, start=40, fold=4).head().round(2).to_csv(
    "inflection-titers-example.csv"
)

df_endpoints = sigmoid.endpoints(cutoff_absolute=0.1, hdi_prob=0.95)
df_endpoints.head().round(2).to_csv("endpoints-example.csv")

# %%

labels = it.index_to_titer(sigmoid.log_dilutions, start=40, fold=4).astype(int)
df_inflections = sigmoid.inflections()
inflection_titer = df_inflections.loc["21-P0609-v001sr01", "mean"]
ymax = sigmoid.posterior["d"].mean()
xmin = -2
xmax = 9
fontsize = 10

plt.figure(figsize=(5, 3))
plt.style.use("seaborn-white")
sigmoid.plot_sample(
    "21-P0609-v001sr01",
    mean=True,
    line_kwds=dict(clip_on=False, c="black"),
    # scatter_kwds=dict(clip_on=False, s=25, lw=0.5, ec="white"),
    scatter_kwds=dict(marker="x", zorder=0, c="grey", clip_on=False),
    xmin=xmin,
    xmax=xmax,
)
for y in 0, ymax, ymax / 2:
    plt.hlines(y, xmin, xmax, colors="lightgrey", clip_on=False)
plt.annotate("Max. response", (xmax, ymax), va="bottom", ha="right", fontsize=fontsize)
plt.annotate(
    "Half max. response", (xmax, ymax / 2), va="bottom", ha="right", fontsize=fontsize
)
plt.annotate(
    "",
    xy=(inflection_titer, 0),
    xytext=(inflection_titer, ymax / 2),
    arrowprops=dict(arrowstyle="->"),
)
plt.annotate(
    "Inflection titer",
    (inflection_titer - 0.1, ymax * 0.25),
    fontsize=fontsize,
    clip_on=False,
    va="center",
    ha="right",
)
plt.ylim(0, ymax + 0.2)
plt.xlim(xmin, xmax)
plt.xticks(sigmoid.log_dilutions, labels, rotation=90)
plt.xlabel("Dilution")
for spine in "top", "right", "bottom", "left":
    plt.gca().spines[spine].set_visible(False)
plt.savefig("sigmoid.png", bbox_inches="tight", dpi=300)
