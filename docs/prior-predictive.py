import numpy as np
import matplotlib.pyplot as plt
import ititer as it


df = it.load_example_data()
df["Log Dilution"] = it.titer_to_index(df["Dilution"], start=40, fold=4)

sigmoid = it.Sigmoid().fit(
    data=df,
    response="OD",
    sample_labels="Sample",
    log_dilution="Log Dilution",
    prior_predictive=True,
)

x = np.linspace(-2.5, 2.5).reshape(-1, 1)
y = it.inverse_logit(
    x=x,
    a=sigmoid.prior_predictive["a"][:, 0],
    b=sigmoid.prior_predictive["b"],
    c=0,
    d=sigmoid.prior_predictive["d"],
)

plt.plot(x, y, c="grey", alpha=0.5, lw=0.5)
plt.xlabel("Log dilution Z-scores")
plt.ylabel("Response")
plt.ylim(0, 3)
plt.xlim(-2.5, 2.5)
plt.savefig("prior-predictive.png", bbox_inches="tight", dpi=300)
