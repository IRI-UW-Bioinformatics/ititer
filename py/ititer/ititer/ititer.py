from numbers import Real
from typing import Union, Iterable, Hashable

import pandas as pd
import numpy as np
import pymc3 as pm


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

        :param log_dilutons: Log diluton values.
        :param response: Response values.
        :param sample_labels: Sample labels.
        :param data: An optional DataFrame. If this is supplied then
        log_dilutions, response, and sample_labels should be columns in the
        DataFrame.
        :param draws: Number of samples to draw from the posterior distribution.
        :param **kwds: Passed to pymc3.sample.
        """
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

        log_dilution_mu = np.mean(log_dilution)
        log_dilution_std = np.std(log_dilution)
        x = (log_dilution - log_dilution_mu) / log_dilution_std

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
            sigmoid.posterior = pm.sample(draws=draws, **kwds)

        return sigmoid

    def plot_fit(self, sample: Union[str, int]):
        """
        Visualise the model fit.
        """
        pass
