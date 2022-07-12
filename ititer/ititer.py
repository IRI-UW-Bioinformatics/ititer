import math
import operator
from os import path
from numbers import Real
from typing import Union, Iterable, Hashable, Generator
import pandas as pd

import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")


class ModelNotFittedError(Exception):
    """
    Raise this warning when a user tries to access attributes only available
    after a model is fit.
    """

    pass


def _contains_null(array) -> bool:
    """
    Test if an array contains nan values.
    """
    return pd.isnull(array).any()


def _batches(iterable: Iterable, n: int) -> Generator[tuple, None, None]:
    """
    Generate batches length n of an iterable.
    """
    iterable = tuple(iterable)
    i = 0
    while i < len(iterable):
        yield iterable[i: i + n]
        i += n


def load_example_data() -> pd.DataFrame:
    """
    Load an example pandas DataFrame to illustrate fitting sigmoid curves.
    """
    return pd.read_csv(path.join(path.dirname(__file__), "..", "data", "od.csv"))


def inverse_logit(
    x: Union[Iterable[Real], Real], a: Real, b: Real, c: Real, d: Real
) -> Union[Iterable, Real]:
    """
    Inverse logit function.

    :param x: x.
    :param a: Location of inflection point on x-axis.
    :param b: Gradient of slope.
    :param c: Bottom y-asymptote.
    :param d: Distance to top y-asymptote from bottom.
    """
    return c + d / (1 + np.exp(b * (x - a)))


def _check_start_fold_valid(start: Real, fold: Real) -> None:
    """
    Check that start and fold will make valid dilution series.

    Helper function used by titer_to_index and index_to_titer.
    """
    if not isinstance(start, Real):
        raise ValueError("start should be a single number")
    if not isinstance(fold, Real):
        raise ValueError("fold should be a single number")
    if start < 0:
        raise ValueError("start must be positive")
    if fold < 0:
        raise ValueError("fold must be positive")


def titer_to_index(
    titer: Union[Iterable[Real], Real], start: Real, fold: Real
) -> Union[Iterable[Real], Real]:
    """
    Log transform a titer / dilution to its position in a dilution series.

    Titers are referred to by their reciprocal, so a titer/dilution of '1/10' is
    referred to as simply '10'.

    A starting dilution of 10 and a fold change of 2 will generate a dilution
    series of: (10, 20, 40, 80, 160, ...). This function takes a dilution, and
    returns its index in the dilution series.

    :param titer: Titer(s) to transform.
    :param start: Starting dilution of the series.
    :param fold: Fold change of the series.
    """
    _check_start_fold_valid(start, fold)
    titer = np.array(titer) if isinstance(titer, (list, tuple)) else titer
    return np.log(titer / start) / np.log(fold)


def index_to_titer(
    index: Union[Iterable[Real], Real], start: Real, fold: Real
) -> Union[Iterable[Real], Real]:
    """
    Transform a position in a dilution series to a titer.

    Titers are referred to by their reciprocal, so a titer/dilution of '1/10' is
    referred to as simply '10'.

    A starting dilution of 10 and a fold change of 2 will generate a dilution
    series of: (10, 20, 40, 80, 160, ...). This function takes a position in
    this dilution series and returns the dilution.

    :param index: Indices(s) to transform.
    :param start: Starting dilution of the series.
    :param fold: Fold change of the series.
    """
    _check_start_fold_valid(start, fold)
    index = np.array(index) if isinstance(index, (list, tuple)) else index
    return start * fold**index


class Sigmoid:
    """
    Sigmoid model for a dose response curve. The model fits a response,
    :math:`y`, as a function of log dilution, :math:`x`:

    :math:`y = c + d / (1 + e^{b(x - a)})`

    Posterior distributions of each parameter can be inferred by either fully or
    partially pooling inference across samples (`partial` vs `full`).
    Parameters can also be fixed *a priori*, by providing a float.

    :param a: ``'partial'``, ``'full'`` or float.
    :param b: ``'partial'``, ``'full'`` or float.
    :param c: ``'partial'``, ``'full'`` or float.
    :param d: ``'partial'``, ``'full'`` or float.
    """

    def __init__(
        self,
        a: Union[str, Real] = "partial",
        b: Union[str, Real] = "full",
        c: Union[str, Real] = "full",
        d: Union[str, Real] = "full",
    ):
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
        data: Union[pd.DataFrame, None] = None,
        draws: int = 10_000,
        prior_predictive: bool = False,
        **kwds,
    ):
        """
        Fit parameters of the sigmoid curve to data.

        :param log_dilution: Log dilution values.
        :param response: Response values.
        :param sample_labels: Sample labels.
        :param data: Optional DataFrame. If supplied then `log_dilutions`,
            `response`, and `sample_labels` should be columns in the DataFrame.
        :param draws: Number of samples to draw from the posterior distribution.
        :param kwds: Passed to :py:func:`pymc3.sample`.
        :param prior_predictive: Sample from the prior predictive distribution.
            The returned Sigmoid object has a `prior_predictive` attribute.
        :returns: Sigmoid object with `posterior` attribute.
        """
        if isinstance(data, pd.DataFrame):
            log_dilution = data[log_dilution].values
            response = data[response].values
            sample_labels = data[sample_labels].values

        elif data is not None:
            raise ValueError("data should be a pandas.DataFrame or None")

        else:
            log_dilution = np.array(log_dilution)
            response = np.array(response)
            sample_labels = np.array(sample_labels)

        if _contains_null(log_dilution):
            raise ValueError("log_dilution contains nan values")

        if _contains_null(response):
            raise ValueError("response contains nan values")

        if _contains_null(sample_labels):
            raise ValueError("sample_labels contains nan values")

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
                a_unit = pm.Normal("a_unit", 0, 1, shape=n_samples)
                a = pm.Deterministic("a", a_unit * sigma_a + mu_a)[sample_idx]
            elif self.a == "full":
                a = pm.Normal("a", 0, 1)
            else:
                a = self.a

            if self.b == "partial":
                mu_b = pm.Normal("mu_b", 0, 1)
                sigma_b = pm.Exponential("sigma_b", 1)
                b_unit = pm.Normal("b_unit", 0, 1, shape=n_samples)
                b = pm.Deterministic("b", b_unit * sigma_b + mu_b)[sample_idx]
            elif self.b == "full":
                b = pm.Normal("b", 0, 1)
            else:
                b = self.b

            if self.c == "partial":
                mu_c = pm.Normal("mu_c", 0, 1)
                sigma_c = pm.Exponential("sigma_c", 1)
                c_unit = pm.Normal("b_unit", 0, 1, shape=n_samples)
                c = pm.Deterministic("c", c_unit * sigma_c + mu_c)[sample_idx]
            elif self.c == "full":
                c = pm.Normal("c", 0, 1)
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
            if prior_predictive:
                sigmoid.prior_predictive = pm.sample_prior_predictive()
            sigmoid.posterior = pm.sample(draws=draws, return_inferencedata=False, **kwds)

        sigmoid.mu_log_dilution = mu_log_dilution
        sigmoid.sd_log_dilution = sd_log_dilution
        sigmoid.sample_i = sample_i
        sigmoid.samples = uniq_samples
        sigmoid.data = pd.DataFrame(
            {
                "log dilution": log_dilution,
                "log dilution std": x,
                "response": response,
                "sample": sample_labels,
            }
        )

        return sigmoid

    def plot_sample(
        self,
        sample: Hashable,
        points: bool = True,
        mean: bool = False,
        scatter_kwds: dict = None,
        line_kwds: dict = None,
        step: int = 1000,
        match_point_color_to_line_color: bool = False,
        xmin: Real = None,
        xmax: Real = None,
    ) -> None:
        """
        Plot sigmoid curves from the posterior distribution of a sample.

        :param sample: The sample to plot.
        :param points: Whether to plot the data as well.
        :param mean: Show the mean of the posterior distribution, rather than samples
            from the posterior.
        :param scatter_kwds: Passed to :py:func:`matplotlib.pyplot.scatter`.
            Used to control the appearance of the data points.
        :param line_kwds: Passed to :py:func:`matplotlib.pyplot.plot`. Used to control the
            appearance of the lines.
        :param step: Show every step'th sample from the posterior. Only has an
            effect if mean is `False`.
        :param match_point_color_to_line_color: Match the color of the data
            points to the lines. This is useful when you want different samples on a
            single plot to have distinct colors, but for the lines and points of one
            sample to match.
        :param xmin: Lowest value on the x-axis to plot. Log dilution units. If
            None, the lowest log dilution in the data is used.
        :param xmax: Highest value on the x-axis to plot. Log dilution units. If
            None, the highest log dilution in the data is used.
        """
        def_scatter_kwds = dict(c="black", zorder=10)
        def_line_kwds = dict(lw=1 if mean else 0.5, zorder=5, c=None if mean else "grey")
        scatter_kwds = (
            def_scatter_kwds
            if scatter_kwds is None
            else {**def_scatter_kwds, **scatter_kwds}
        )
        line_kwds = def_line_kwds if line_kwds is None else {**def_line_kwds, **line_kwds}
        i = self.sample_i[sample]
        xmin = self.data["log dilution std"].min() if xmin is None else self.scale(xmin)
        xmax = self.data["log dilution std"].max() if xmax is None else self.scale(xmax)
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
        lines = plt.plot(self.inverse_scale(xgrid), ygrid, **line_kwds)

        if points:
            sub_df = self.data.set_index("sample").loc[sample]
            if match_point_color_to_line_color:
                scatter_kwds["c"] = lines[0].get_color()
            plt.scatter(
                sub_df["log dilution"],
                sub_df["response"],
                **scatter_kwds,
            )

        plt.xticks(self.log_dilutions)
        plt.xlabel("Log dilution")
        plt.ylabel("Response")

    def plot_samples(self, samples: Iterable[Hashable], **kwds) -> None:
        """
        Plot sigmoid curves of samples using the mean value of the posterior distribution.

        :param samples: List of samples to plot.
        :param kwds: Passed to :py:meth:`Sigmoid.plot_sample`.
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

    def plot_all_samples(
        self,
        samples_per_ax: int = 9,
        n_cols: int = 4,
        ax_width: Real = 7,
        ax_height: Real = 4,
    ) -> None:
        """
        Plot sigmoid curves for all samples using the mean posterior value of
        each parameter.

        :param samples_per_ax: Number of samples to put on a single ax.
        :param n_cols: Number of columns of axes. The number of rows is computed based
            on this and samples_per_ax.
        :param ax_width: Width of a single ax.
        :param ax_height: Height of a single ax.
        """
        n_rows = math.ceil(len(self.samples) / samples_per_ax / n_cols)

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            sharex=True,
            sharey=True,
            figsize=(n_cols * ax_width, n_rows * ax_height),
        )

        batches = _batches(self.samples, samples_per_ax)

        for samples, ax in zip(batches, fig.axes):
            plt.sca(ax)
            self.plot_samples(samples)

    def inflections(self, hdi_prob: float = 0.95) -> pd.DataFrame:
        """
        Summarise the posterior distribution of inflection points for each sample.

        The returned DataFrame has these columns:

            - `mean`: Mean value.
            - `median`: Median value.
            - `hdi low`: Lower boundary of the highest density interval (HDI).
            - `hdi high`: Upper boundary of the HDI.

        :param hdi_prob: The width of the HDI to calculate.
        """
        try:
            # Calling pm.hdi issues warnings about not being in a model context
            # better to explicitly make an arviz inference data object
            # az.convert_to_inference_data expects dimensions of (chains, draws,
            # shape), hence np.newaxis
            i_data = az.convert_to_inference_data(self.posterior["a"][np.newaxis])
        except AttributeError:
            raise ModelNotFittedError

        hdi = pm.hdi(i_data, hdi_prob=hdi_prob).x

        df = pd.DataFrame(
            {
                "mean": np.mean(self.posterior["a"], axis=0),
                "median": np.median(self.posterior["a"], axis=0),
                "hdi low": hdi[:, 0].values,
                "hdi high": hdi[:, 1].values,
            },
            index=self.samples,
        )

        df.index.name = "sample"

        return self.inverse_scale(df)

    def endpoints(
        self,
        cutoff_proportion: Union[Real, None] = None,
        cutoff_absolute: Union[Real, None] = None,
        hdi_prob: Real = 0.95,
    ) -> pd.DataFrame:
        """
        Compute endpoints for each sample, given some response. An endpoint is
        the dilution at which a particular value of the response is obtained,
        known as the cut-off. The cut-off is either in absolute units, or given
        as a proportion of `d`.

        Must supply exactly one of either `cutoff_proportion` or `cutoff_absolute`.

        The returned DataFrame contains endpoints on the log-transformed scale.

        :param cutoff_proportion: Proportion of `d`. Must be in interval (0, 1).
        :param cutoff_absolute: Absolute value of `d`.
        """
        if not operator.xor(cutoff_absolute is None, cutoff_proportion is None):
            raise ValueError("Must give either cutoff_absolute or cutoff_proportion")

        if cutoff_proportion is not None:
            if not isinstance(cutoff_proportion, Real):
                raise ValueError("cutoff_proportion must be a number")
            if not (0 < cutoff_proportion < 1):
                raise ValueError("cutoff_proportion must be in interval (0, 1)")
            elif self.d == "partial":
                # When d has per-sample estimates, y also gets per-sample values
                y = self.posterior["d"].mean(axis=0) * cutoff_proportion
            elif self.d == "full":
                y = self.posterior["d"].mean() * cutoff_proportion
            else:
                y = self.d * cutoff_proportion

        else:
            if not isinstance(cutoff_absolute, Real):
                raise ValueError("cutoff_absolute must be a number")
            else:
                y = cutoff_absolute

        # Get posterior distributions of parameter values in correct shape
        params = {}
        for param in "abcd":
            value = getattr(self, param)
            if value == "full":
                params[param] = self.posterior[param][:, np.newaxis]
            elif value == "partial":
                params[param] = self.posterior[param]
            else:
                params[param] = value

        def f(y, a, b, c, d):
            return a + (np.log((d / (y - c)) - 1) / b)

        posterior_endpoints = f(y, **params)

        # Calling pm.hdi issues warnings about not being in a model context
        # better to explicitly make an arviz inference data object
        # az.convert_to_inference_data expects dimensions of (chains, draws,
        # shape), hence np.newaxis
        i_data = az.convert_to_inference_data(posterior_endpoints[np.newaxis])

        hdi = pm.hdi(i_data, hdi_prob=hdi_prob).x

        df = pd.DataFrame(
            {
                "mean": np.mean(posterior_endpoints, axis=0),
                "median": np.median(posterior_endpoints, axis=0),
                "hdi low": hdi[:, 0].values,
                "hdi high": hdi[:, 1].values,
            },
            index=self.samples,
        )

        df.index.name = "sample"

        return self.inverse_scale(df)

    def scale(self, x: Union[Real, np.ndarray]) -> Union[Real, np.ndarray]:
        """
        Log dilutions are scaled to have mean of 0 and standard deviation of 1
        for efficient inference. Apply the same scaling to `x`, i.e. from the log
        dilution scale to the standardised log dilution scale.

        :param x: Value(s) to scale.
        """
        return (x - self.mu_log_dilution) / self.sd_log_dilution

    def inverse_scale(self, x: Union[Real, np.ndarray]) -> Union[Real, np.ndarray]:
        """
        Log dilutions are scaled to have mean of 0 and standard deviation of 1
        for efficient inference. Apply the inverse scaling to `x`, i.e. from the
        standardised log dilution scale back to the log dilution scale.

        :param x: Value(s) to scale.
        """
        return (x * self.sd_log_dilution) + self.mu_log_dilution

    @property
    def log_dilutions(self) -> np.array:
        """
        Sorted array of unique log dilutions in the data.
        """
        try:
            return np.sort(self.data["log dilution"].unique())
        except AttributeError:
            raise ModelNotFittedError(
                "An unfitted Sigmoid has no log dilutions. Call Sigmoid.fit first."
            )

    @property
    def log_dilutions_std(self) -> np.array:
        """
        Sorted array of unique standardised log dilutions in the data.
        """
        try:
            return self.scale(self.log_dilutions)
        except ModelNotFittedError:
            raise ModelNotFittedError(
                "An unfitted Sigmoid has no standardised log dilutions. Call Sigmoid.fit first."
            )
