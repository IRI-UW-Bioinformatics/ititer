import unittest
import pymc3 as pm
from ititer import Sigmoid


class TestSigmoid(unittest.TestCase):
    """
    Tests for ititer.Sigmoid
    """

    def test_a_partial_by_default(self):
        """
        a parameter should be partially pooled by default.
        """
        sigmoid = Sigmoid()
        self.assertEqual("partial", sigmoid.a)

    def test_b_full_by_default(self):
        """
        b parameter should be fully pooled by default.
        """
        sigmoid = Sigmoid()
        self.assertEqual("full", sigmoid.b)

    def test_c_full_by_default(self):
        """
        c parameter should be fully pooled by default.
        """
        sigmoid = Sigmoid()
        self.assertEqual("full", sigmoid.c)

    def test_d_full_by_default(self):
        """
        d parameter should be fully pooled by default.
        """
        sigmoid = Sigmoid()
        self.assertEqual("full", sigmoid.d)

    def test_passing_list_to_a_raises_value_error(self):
        """
        A value error should be raised if a is not one of
        ['partial', 'full', numbers.Real, pymc3.distribution].
        """
        with self.assertRaisesRegex(
            ValueError, "a should be 'partial', 'full' or a number."
        ):
            Sigmoid(a=[])

    def test_passing_unrecognised_string(self):
        """
        Passing a string that is not 'partial' or 'full' should
        raise a ValueError.
        """
        msg = "Only strings 'partial' and 'full' allowed"
        with self.assertRaisesRegex(ValueError, msg):
            Sigmoid(a="abc")

    def test_passing_multidimensional_log_dilution(self):
        """
        Passing multidimensional arrays to Sigmoid.fit for log_dilution,
        should raise a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "log_dilution not 1 dimensional"):
            Sigmoid().fit(
                log_dilution=[
                    [1, 2, 3, 4, 1, 2, 3, 4],
                ],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            )

    def test_passing_multidimensional_response(self):
        """
        Passing multidimensional arrays to Sigmoid.fit for response should raise
        a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "response not 1 dimensional"):
            Sigmoid().fit(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[
                    [1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                ],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            )

    def test_passing_multidimensional_sample_labels(self):
        """
        Passing multidimensional arrays to Sigmoid.fit for sample_lables should
        raise a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "sample_labels not 1 dimensional"):
            Sigmoid().fit(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=(("a", "a", "a", "a", "b", "b", "b", "b"),),
            )

    def test_passing_length_mismatch(self):
        """
        Passing mismatched length inputs should raise a ValueError.
        """
        with self.assertRaisesRegex(
            ValueError,
            "log_dilution \(8\), response \(7\) and sample_labels \(8\) not "
            "the same length",
        ):
            Sigmoid().fit(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            )


class TestSigmoidSampling(unittest.TestCase):
    """
    Tests for ititer.Sigmoid that involve sampling from the posterior
    distribution using pymc3. These tests are much slower as each has to enter a
    pymc3 model context. (The sampling itself is not the slow part.)

    These tests can be skipped by on the command line by doing:

        $ pytest -k 'not TestSigmoidSampling'
    """

    def test_fit_returns_sigmoid(self):
        """
        Sigmoid.fit should return a new Sigmoid instance.
        """
        a = Sigmoid(a="full")
        b = a.fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertIsInstance(
            b, Sigmoid, "Sigmoid.fit does not return an instance of Sigmoid"
        )
        self.assertNotEqual(
            id(a), id(b), "Sigmoid.fit does not return a new Sigmoid instance"
        )

    def test_fit_attaches_posterior(self):
        """
        The object returned by Sigmoid.fit should have an attribute called
        posterior which is a pymc3 MultiTrace instance.
        """
        a = Sigmoid(a="full")
        self.assertFalse(
            hasattr(a, "posterior"),
            "Fresh Sigmoid objects shouldn't have a posterior attribute",
        )
        b = a.fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertTrue(
            hasattr(b, "posterior"),
            "Sigmoid objects returned by fit should have a posterior attribute",
        )
        self.assertIsInstance(b.posterior, pm.backends.base.MultiTrace)

    def test_a_partial_variables(self):
        """
        When a='partial' expect variables mu_a, sigma_a and a.
        """
        sigmoid = Sigmoid(a="partial").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertIn("mu_a", sigmoid.posterior.varnames)
        self.assertIn("sigma_a", sigmoid.posterior.varnames)
        self.assertIn("a", sigmoid.posterior.varnames)

    def test_a_full_variables(self):
        """
        When a='full' expect only the variable a, and not mu_a or sigma_a.
        """
        sigmoid = Sigmoid(a="full").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_a", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_a", sigmoid.posterior.varnames)
        self.assertIn("a", sigmoid.posterior.varnames)

    def test_a_const_variables(self):
        """
        When a is a constant neither a, mu_a nor sigma_a should be in posterior.
        """
        sigmoid = Sigmoid(a=1).fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_a", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_a", sigmoid.posterior.varnames)
        self.assertNotIn("a", sigmoid.posterior.varnames)

    def test_b_partial_variables(self):
        """
        When b='partial' expect variables mu_b, sigma_b and b.
        """
        sigmoid = Sigmoid(b="partial").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertIn("mu_b", sigmoid.posterior.varnames)
        self.assertIn("sigma_b", sigmoid.posterior.varnames)
        self.assertIn("b", sigmoid.posterior.varnames)

    def test_b_full_variables(self):
        """
        When b='full' expect variables mu_b, sigma_b and b.
        """
        sigmoid = Sigmoid(b="full").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_b", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_b", sigmoid.posterior.varnames)
        self.assertIn("b", sigmoid.posterior.varnames)

    def test_b_const_variables(self):
        """
        When b is a constant neither b, mu_b nor sigma_b should be in posterior.
        """
        sigmoid = Sigmoid(b=1).fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_b", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_b", sigmoid.posterior.varnames)
        self.assertNotIn("b", sigmoid.posterior.varnames)

    def test_c_partial_variables(self):
        """
        When c='partial' expect variables mu_c, sigma_c and c.
        """
        sigmoid = Sigmoid(c="partial").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertIn("mu_c", sigmoid.posterior.varnames)
        self.assertIn("sigma_c", sigmoid.posterior.varnames)
        self.assertIn("c", sigmoid.posterior.varnames)

    def test_c_full_variables(self):
        """
        When c='full' expect variables mu_c, sigma_c and c.
        """
        sigmoid = Sigmoid(c="full").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_c", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_c", sigmoid.posterior.varnames)
        self.assertIn("c", sigmoid.posterior.varnames)

    def test_c_const_variables(self):
        """
        When c is a constant neither c, mu_c nor sigma_c should be in posterior.
        """
        sigmoid = Sigmoid(c=1).fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_c", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_c", sigmoid.posterior.varnames)
        self.assertNotIn("c", sigmoid.posterior.varnames)

    def test_d_partial_variables(self):
        """
        When d='partial' expect variables sigma_d and d. d is constrained to be
        positive, thus there is no mu_d.
        """
        sigmoid = Sigmoid(d="partial").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_d", sigmoid.posterior.varnames)
        self.assertIn("sigma_d", sigmoid.posterior.varnames)
        self.assertIn("d", sigmoid.posterior.varnames)

    def test_d_full_variables(self):
        """
        When d='full' expect only the variable d.
        """
        sigmoid = Sigmoid(d="full").fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_d", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_d", sigmoid.posterior.varnames)
        self.assertIn("d", sigmoid.posterior.varnames)

    def test_d_const_variables(self):
        """
        When d is a constant neither d, mu_d nor sigma_d should be in posterior.
        """
        sigmoid = Sigmoid(d=1).fit(
            log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
            response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
            sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            draws=2,
        )
        self.assertNotIn("mu_d", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_d", sigmoid.posterior.varnames)
        self.assertNotIn("d", sigmoid.posterior.varnames)


if __name__ == "__main__":
    unittest.main()
