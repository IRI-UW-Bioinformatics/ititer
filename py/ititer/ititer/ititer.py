from os import path
from numbers import Real
from typing import Union, Iterable, Hashable

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt


class ModelNotFittedWarning(Exception):
    """
    Raise this warning when a user tries to access attributes only available
    after a model is fit.
    """

    pass


def load_example_data() -> pd.DataFrame:
    """
    Load an example pandas DataFrame to illustrate fitting sigmoid curves.
    """
    return pd.read_csv(path.join(path.dirname(__file__), "..", "data", "od.csv"))


def inverse_logit(
    x: Union[Iterable[Real], Real], a: Real, b: Real, c: Real, d: Real
) -> Union[Iterable, Real]:
    """
    Inverse logit function at point x.

    :param x: x.
    :param a: Location of inflection point on x-axis.
    :param b: Gradient of slope.
    :param c: Bottom y-asymptote.
    :param d: Distance to top y-asymptote from bottom.
    """
    return c + d / (1 + np.exp(b * (x - a)))


def log_transform_titer(
    titer: Union[Iterable[Real], Real], start: Real, fold: Real
) -> Union[Iterable[Real], Real]:
    """
    Log transform a titer.

    :param titer: A titer or titers to transform.
    :param start: The starting dilution of the dilution series.
    :param fold: The fold change in concentration at each step in the dilution
    series.
    :returns: numpy.ndarray if titer is iterable, otherwise the same type as
    titer.
    """
    if not isinstance(start, Real):
        raise ValueError("start should be a single number")
    if not isinstance(fold, Real):
        raise ValueError("fold should be a single number")
    if start < 0:
        raise ValueError("start must be positive")
    if fold < 0:
        raise ValueError("fold must be positive")
    titer = np.array(titer) if isinstance(titer, Iterable) else titer
    return np.log(titer / start) / np.log(fold)


class Sigmoid:
    def __init__(
        self,
        a: Union[str, Real] = "partial",
        b: Union[str, Real] = "full",
        c: Union[str, Real] = "full",
        d: Union[str, Real] = "full",
    ):
        """
        Sigmoid model for a dose response curve.

        :param a:
        :param b:
        :param c:
        :param d:
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        for label in "a", "b", "c", "d":
            value = getattr(self, label)

            if not isinstance(value, (str, Real)):
                msg = "{} should be 'partial', 'full' or a number. Passed '{}']"
                raise ValueError(msg.format(label, value))

            if isinstance(value, str) and value.lower() not in {"partial", "full"}:
                msg = "Only strings 'partial' and 'full' allowed. Passed '{}'"
                raise ValueError(msg.format(label))

    def fit(
        self,
        log_dilution: Union[Iterable[Real], Hashable],
        response: Union[Iterable[Real], Hashable],
        sample_labels: Union[Iterable[Hashable], Hashable],
        data: pd.DataFrame = None,
        draws: int = 10_000,
        **kwds,
    ):
        """
        Fit parameters of the sigmoid curve to data.

        After this method is run the Sigmoid object has a posterior attribute.

        :param log_diluton: Log diluton values.
        :param response: Response values.
        :param sample_labels: Sample labels.
        :param data: An optional DataFrame. If this is supplied then
        log_dilutions, response, and sample_labels should be columns in the
        DataFrame.
        :param draws: Number of samples to draw from the posterior distribution.
        :param **kwds: Passed to pymc3.sample.
        """
        if isinstance(data, pd.DataFrame):
            log_dilution = data[log_dilution]
            response = data[response]
            sample_labels = data[sample_labels]
        elif data is not None:
            raise ValueError("data should be a pandas DataFrame or None")
        else:
            log_dilution = np.array(log_dilution)
            response = np.array(response)
            sample_labels = np.array(sample_labels)

        if 1 != log_dilution.ndim:
            raise ValueError("log_dilution not 1 dimensional")

        if 1 != response.ndim:
            raise ValueError("response not 1 dimensional")

        if 1 != sample_labels.ndim:
            raise ValueError("sample_labels not 1 dimensional")

        if len(log_dilution) != len(response) != len(sample_labels):
            raise ValueError(
                "log_dilution ({}), response ({}) and sample_labels ({}) "
                "not the same length".format(
                    len(log_dilution), len(response), len(sample_labels)
                )
            )

        sigmoid = Sigmoid(self.a, self.b, self.c, self.d)

        uniq_samples = tuple(set(sample_labels))
        n_samples = len(uniq_samples)
        sample_idx = np.array([uniq_samples.index(sample) for sample in sample_labels])
        sample_i = {sample: uniq_samples.index(sample) for sample in sample_labels}

        mu_log_dilution = np.mean(log_dilution)
        sd_log_dilution = np.std(log_dilution)
        x = (log_dilution - mu_log_dilution) / sd_log_dilution

        with pm.Model() as self.model:

            if self.a == "partial":
                mu_a = pm.Normal("mu_a", 0, 1)
                sigma_a = pm.Exponential("sigma_a", 1)
                a = pm.Normal("a", mu_a, sigma_a, shape=n_samples)[sample_idx]
            elif self.a == "full":
                a = pm.Normal("a", 0, 1)
            else:
                a = self.a

            if self.b == "partial":
                mu_b = pm.Normal("mu_b", 0, 1)
                sigma_b = pm.Exponential("sigma_b", 1)
                b = pm.Normal("b", mu_b, sigma_b, shape=n_samples)[sample_idx]
            elif self.b == "full":
                b = pm.Normal("b", 0, 1)
            else:
                b = self.b

            if self.c == "partial":
                mu_c = pm.Normal("mu_c", 0, 1)
                sigma_c = pm.Exponential("sigma_c", 1)
                c = pm.Normal("c", mu_c, sigma_c, shape=n_samples)[sample_idx]
            elif self.c == "full":
                c = pm.Normal("c", 0, 0.05)
            else:
                c = self.c

            if self.d == "partial":
                sigma_d = pm.Exponential("sigma_d", 1)
                d = pm.Exponential("d", sigma_d, shape=n_samples)[sample_idx]
            elif self.d == "full":
                d = pm.Exponential("d", 1)
            else:
                d = self.d

            sigma = pm.Exponential("sigma", 1)
            mu = c + d * pm.math.invlogit(-b * (x - a))
            pm.Normal("lik", mu, sigma, observed=response)
            sigmoid.posterior = pm.sample(
                draws=draws, return_inferencedata=False, **kwds
            )

        sigmoid.mu_log_dilution = mu_log_dilution
        sigmoid.sd_log_dilution = sd_log_dilution
        sigmoid.sample_i = sample_i
        sigmoid.samples = list(sample_i)
        sigmoid.data = pd.DataFrame(
            {
                "log dilution": log_dilution,
                "log dilution std": x,
                "response": response,
                "sample labels": sample_labels,
            }
        )

        return sigmoid

    def plot_sample(
        self,
        sample: Hashable,
        points: bool = True,
        mean: bool = False,
        scatter_kwds=None,
        line_kwds=None,
        step: int = 1000,
        match_point_color_to_line_color: bool = False,
    ) -> None:
        """
        Plot sigmoid curves from the posterior distribution of a sample.

        :param sample: The sample to plot.
        :param points: Whether to plot the data as well.
        :param mean: Show the mean of the posterior distribution, rather than samples
        from the posterior.
        :param scatter_kwds: Passed to matplotlib.pyplot.scatter. Used to control the
        appearance of the data points.
        :param line_kwds: Passed to matplotlib.pyplot.plot. Used to control the
        appearance of the lines.
        :param step: Show every step'th sample from the posterior. Only has an
        effect if mean is False.
        :param match_point_color_to_line_color: Match the color of the data
        points to the lines. This is useful when you want different samples on a
        single plot to have distinct colors, but for the lines and points of one
        sample to match.
        """
        def_scatter_kwds = dict(c="black", zorder=10)
        def_line_kwds = dict(
            lw=1 if mean else 0.5, zorder=5, c=None if mean else "grey"
        )
        scatter_kwds = (
            def_scatter_kwds
            if scatter_kwds is None
            else {**def_scatter_kwds, **scatter_kwds}
        )
        line_kwds = (
            def_line_kwds if line_kwds is None else {**def_line_kwds, **line_kwds}
        )
        i = self.sample_i[sample]
        xmin = self.data["log dilution std"].min()
        xmax = self.data["log dilution std"].max()
        xgrid = np.linspace(xmin, xmax).reshape(-1, 1)
        params = {}
        step = 1 if mean else step
        for param in "a", "b", "c", "d":
            value = getattr(self, param)
            if value == "partial":
                params[param] = self.posterior[param][::step, i]
            elif value == "full":
                params[param] = self.posterior[param][::step]
            else:
                params[param] = value
        ygrid = inverse_logit(xgrid, **params)
        ygrid = ygrid.mean(axis=1) if mean else ygrid
        lines = plt.plot(xgrid, ygrid, **line_kwds)

        if points:
            sub_df = self.data.set_index("sample labels").loc[sample]
            if match_point_color_to_line_color:
                scatter_kwds["c"] = lines[0].get_color()
            plt.scatter(
                sub_df["log dilution std"],
                sub_df["response"],
                **scatter_kwds,
            )

        plt.xticks(ticks=self.log_dilutions_std, labels=self.log_dilutions)
        plt.xlabel("Log dilution")
        plt.ylabel("Response")

    def plot_samples(self, samples: Iterable[Hashable], **kwds) -> None:
        """
        Plot sigmoid curves of samples using the mean value of the posterior distribution.

        :param samples: The samples to plot.
        :param kwds: Passed to Sigmoid.plot_sample.
        """
        line_kwds = kwds.pop("line_kwds", {})

        for sample in samples:
            self.plot_sample(
                sample,
                mean=True,
                line_kwds={**dict(label=sample), **line_kwds},
                match_point_color_to_line_color=True,
                **kwds,
            )
        plt.legend(title="Sample")

    @property
    def log_dilutions(self) -> np.array:
        """
        A sorted array of unique log dilutions in the data.
        """
        try:
            return np.sort(self.data["log dilution"].unique())
        except AttributeError:
            raise ModelNotFittedWarning(
                "An unfitted Sigmoid has no log dilutions. Call Sigmoid.fit first."
            )

    @property
    def log_dilutions_std(self) -> np.array:
        """
        A sorted array of unique standardised log dilutions in the data.
        """
        try:
            return (self.log_dilutions - self.mu_log_dilution) / self.sd_log_dilution
        except ModelNotFittedWarning:
            raise ModelNotFittedWarning(
                "An unfitted Sigmoid has no standardised log dilutions. Call Sigmoid.fit first."
            )
