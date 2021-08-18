#!/usr/bin/env python3

import matplotlib.pyplot as plt
import ititer as it


df = it.load_example_data()
df.head(10).round(3).to_csv("long-example.csv")

wide = df.head(24).pivot(columns="Sample", values="OD", index="Dilution").T
wide.columns.name = None
wide.round(3).to_csv("wide-example.csv")

df["Log Dilution"] = it.titer_to_index(df["Dilution"], start=40, fold=4)

sigmoid = it.Sigmoid(a="partial", b="full", c=0, d="full")

posterior = sigmoid.fit(
    log_dilution=df["Log Dilution"],
    response=df["OD"],
    sample_labels=df["Sample"],
)

posterior.plot_sample("21-P0004-v001sr01", step=1000)
plt.savefig("1-sample.png", bbox_inches="tight")
plt.close()

posterior.plot_sample("21-P0004-v001sr01", mean=True)
plt.savefig("1-sample-mean.png", bbox_inches="tight")
plt.close()

posterior.plot_samples(["21-P0833-v001sr01", "21-P0834-v001sr01"])
plt.savefig("2-samples.png", bbox_inches="tight")
plt.close()

posterior.plot_all_samples()
plt.savefig("all-samples.png", bbox_inches="tight")
plt.close()

df_inflections = posterior.inflections(hdi_prob=0.95)
df_inflections.head().round(2).to_csv("inflections-example.csv")

it.index_to_titer(df_inflections, start=40, fold=4).head().round(2).to_csv(
    "inflection-titers-example.csv"
)

df_endpoints = posterior.endpoints(cutoff_absolute=0.1, hdi_prob=0.95)
df_endpoints.head().round(2).to_csv("endpoints-example.csv")

posterior.samples

# %%

plt.figure(figsize=(5, 3))
plt.style.use("seaborn-whitegrid")
posterior.plot_sample("21-P0609-v001sr01", mean=True, line_kwds=dict(c="black"))
plt.savefig("sigmoid.png", bbox_inches="tight", dpi=200)
