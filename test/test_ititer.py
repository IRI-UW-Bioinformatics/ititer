import unittest
import pymc as pm
import pandas as pd

import ititer as it
from ititer import Sigmoid

FIT_KWDS = dict(tune=1, draws=1)


class TestFittedSigmoid(unittest.TestCase):
    """
    Tests for ititer.Sigmoid after a model has been fit.
    """

    @classmethod
    def setUpClass(cls):
        """
        Fit some data once.
        """
        sigmoid = Sigmoid()
        df = it.load_example_data().head(50)
        df["Log Dilution"] = it.titer_to_index(df["Dilution"], start=40, fold=4)
        cls.sigmoid = sigmoid.fit(
            log_dilution=df["Log Dilution"],
            response=df["OD"],
            sample_labels=df["Sample"],
            **FIT_KWDS
        )

    def test_inflections_dataframe_columns(self):
        """
        Check the inflections DataFrame has the expected columns.
        """
        df_inflections = self.sigmoid.inflections(hdi_prob=0.95)
        self.assertEqual(
            {"mean", "median", "hdi low", "hdi high"}, set(df_inflections.columns)
        )

    def test_inflections_dataframe_shape(self):
        """
        Check the inflections DataFrame has the expected shape.
        """
        df_inflections = self.sigmoid.inflections(hdi_prob=0.95)
        n_samples = len(self.sigmoid.data["sample"].unique())
        self.assertEqual((n_samples, 4), df_inflections.shape)

    def test_must_provide_one_response_to_endpoints(self):
        """
        Must provide exactly one of cutoff_absolute and cutoff_proportion to
        Sigmoid.endpoints.
        """
        msg = "Must give either cutoff_absolute or cutoff_proportion"

        with self.assertRaisesRegex(ValueError, msg):
            self.sigmoid.endpoints()

        with self.assertRaisesRegex(ValueError, msg):
            self.sigmoid.endpoints(cutoff_absolute=0.5, cutoff_proportion=0.5)

    def test_cutoff_proportion_negative(self):
        """
        Providing a negative cutoff_proportion should raise a ValueError.
        """
        msg = r"cutoff_proportion must be in interval \(0, 1\)"
        with self.assertRaisesRegex(ValueError, msg):
            self.sigmoid.endpoints(cutoff_proportion=-0.5)

    def test_cutoff_proportion_gt_1(self):
        """
        Providing cutoff_proportion greater than 1 should raise a ValueError.
        """
        msg = r"cutoff_proportion must be in interval \(0, 1\)"
        with self.assertRaisesRegex(ValueError, msg):
            self.sigmoid.endpoints(cutoff_proportion=1.5)

    def test_endpoints_dataframe_columns(self):
        """
        Check the endpoints DataFrame has the expected columns.
        """
        df_endpoints = self.sigmoid.endpoints(cutoff_absolute=0.5, hdi_prob=0.95)
        self.assertEqual(
            {"mean", "median", "hdi low", "hdi high"}, set(df_endpoints.columns)
        )

    def test_endpoints_dataframe_shape(self):
        """
        Check the endpoints DataFrame has the expected shape.
        """
        df_endpoints = self.sigmoid.endpoints(cutoff_absolute=0.5, hdi_prob=0.95)
        n_samples = len(self.sigmoid.data["sample"].unique())
        self.assertEqual((n_samples, 4), df_endpoints.shape)


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
                **FIT_KWDS
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
                **FIT_KWDS
            )

    def test_passing_multidimensional_sample_labels(self):
        """
        Passing multidimensional arrays to Sigmoid.fit for sample_labels should
        raise a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "sample_labels not 1 dimensional"):
            Sigmoid().fit(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=(("a", "a", "a", "a", "b", "b", "b", "b"),),
                **FIT_KWDS
            )

    def test_passing_length_mismatch(self):
        """
        Passing mismatched length inputs should raise a ValueError.
        """
        with self.assertRaisesRegex(
            ValueError,
            r"log_dilution \(8\), response \(7\) and sample_labels \(8\) not "
            "the same length",
        ):
            Sigmoid().fit(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
                **FIT_KWDS
            )

    def test_passing_column_names(self):
        """
        Should be able to pass a data keyword argument and names of columns in a
        DataFrame.
        """
        df = pd.DataFrame(
            dict(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            )
        )
        Sigmoid().fit(
            data=df,
            response="response",
            sample_labels="sample_labels",
            log_dilution="log_dilution",
            **FIT_KWDS
        )

    def test_cant_pass_data_containing_nan_log_dilution(self):
        """
        Passing data containing nan values should raise ValueError.
        """
        df = pd.DataFrame(
            dict(
                log_dilution=[1, 2, None, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            )
        )
        with self.assertRaisesRegex(ValueError, "log_dilution contains nan values"):
            Sigmoid().fit(
                data=df,
                response="response",
                sample_labels="sample_labels",
                log_dilution="log_dilution",
                **FIT_KWDS
            )

    def test_cant_pass_data_containing_nan_response(self):
        """
        Passing data containing nan values should raise ValueError.
        """
        df = pd.DataFrame(
            dict(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, None, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=["a", "a", "a", "a", "b", "b", "b", "b"],
            )
        )
        with self.assertRaisesRegex(ValueError, "response contains nan values"):
            Sigmoid().fit(
                data=df,
                response="response",
                sample_labels="sample_labels",
                log_dilution="log_dilution",
                **FIT_KWDS
            )

    def test_cant_pass_data_containing_nan_sample_labels(self):
        """
        Passing data containing nan values should raise ValueError.
        """
        df = pd.DataFrame(
            dict(
                log_dilution=[1, 2, 3, 4, 1, 2, 3, 4],
                response=[1, 0.7, 0.3, 0, 1, 0.7, 0.3, 0],
                sample_labels=["a", None, "a", "a", "b", "b", "b", "b"],
            )
        )
        with self.assertRaisesRegex(ValueError, "sample_labels contains nan values"):
            Sigmoid().fit(
                data=df,
                response="response",
                sample_labels="sample_labels",
                log_dilution="log_dilution",
                **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
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
            **FIT_KWDS
        )
        self.assertNotIn("mu_d", sigmoid.posterior.varnames)
        self.assertNotIn("sigma_d", sigmoid.posterior.varnames)
        self.assertNotIn("d", sigmoid.posterior.varnames)


class TestTestDf(unittest.TestCase):
    """
    Tests for the example dataset.
    """

    def test_is_dataframe(self):
        """
        ititer.load_example_data() should return a pandas DataFrame.
        """
        self.assertIsInstance(it.load_example_data(), pd.DataFrame)

    def test_columns(self):
        """
        Test that OD, Sample and Dilution columns are present.
        """
        self.assertEqual(
            {"OD", "Sample", "Dilution"}, set(it.load_example_data().columns)
        )

    def test_shape(self):
        """
        The DataFrame should have 85 rows and 3 columns.
        """
        self.assertEqual((1296, 3), it.load_example_data().shape)

    def test_samples(self):
        """
        There should be 162 unique samples.
        """
        self.assertEqual(162, len(it.load_example_data()["Sample"].unique()))


class TestTiterToIndex(unittest.TestCase):
    """
    Tests for the it.titer_to_index helper function.
    """

    def test_start_eq_titer(self):
        """
        If titer is the same as start, result should be 0, regardless of fold.
        """
        self.assertEqual(0, it.titer_to_index(titer=40, start=40, fold=4))
        self.assertEqual(0, it.titer_to_index(titer=40, start=40, fold=2))

    def test_case_a(self):
        """
        Start = 10, titer = 40, fold = 2 should give index of 2.
        """
        self.assertEqual(2, it.titer_to_index(titer=40, start=10, fold=2))

    def test_negative_start_not_allowed(self):
        """
        Passing a negative start should raise a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "start must be positive"):
            it.titer_to_index(40, -10, 4)

    def test_negative_fold_not_allowed(self):
        """
        Passing a negative fold should raise a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "fold must be positive"):
            it.titer_to_index(40, 40, -1)


class TestIndexToTiter(unittest.TestCase):
    """
    Tests for it.index_to_titer.
    """

    def test_case_a(self):
        """
        Start = 10, index = 2, fold = 2 should give titer of 40.
        """
        self.assertEqual(40, it.index_to_titer(index=2, start=10, fold=2))

    def test_index_to_titer_inverse_of_titer_to_index(self):
        """
        index_to_titer should be the inverse function of titer_to_index.
        """
        start = 3.2
        fold = 5.5
        index = 5
        titer = it.index_to_titer(index=index, start=start, fold=fold)
        self.assertEqual(index, it.titer_to_index(titer=titer, start=start, fold=fold))


class TestBatches(unittest.TestCase):
    """
    Tests for ititer._batches
    """

    def test_yields_tuples(self):
        """
        It should yield tuples.
        """
        batch = next(it._batches("ABCDE", 5))
        self.assertIsInstance(batch, tuple)

    def test_n_matches_length(self):
        """
        If n is the same length as the iterable.
        """
        batches = it._batches("ABCDE", 5)
        self.assertEqual(5, len(next(batches)))

    def test_iterable_multiple_of_n(self):
        """
        Test when the length of the iterator is a multiple of n.
        """
        batches = it._batches("ABCD", 2)
        batch1 = next(batches)
        self.assertEqual(("A", "B"), batch1)
        batch2 = next(batches)
        self.assertEqual(("C", "D"), batch2)
        with self.assertRaises(StopIteration):
            next(batches)

    def test_iterable_not_multiple_of_n(self):
        """
        Test when the length of the iterator is not a multiple of n.
        """
        batches = it._batches("ABC", 2)
        batch1 = next(batches)
        self.assertEqual(("A", "B"), batch1)
        batch2 = next(batches)
        self.assertEqual(("C",), batch2)


if __name__ == "__main__":
    unittest.main()
