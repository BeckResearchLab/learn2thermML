import pytest
import numpy as np
import l2tml_utils.data_utils



class TestBalanceFn:

    def test_double_upsample(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils.get_balance_data_sizes(n_a_i, n_b_i, desired_balance=0.5, max_upsampling=1.0)
        assert np.isclose(n_a_f, 4), "Did not upsample to 4"
        assert np.isclose(n_b_f, 4), "b should be downsampled to 4"

    def test_all_downsample(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils.get_balance_data_sizes(n_a_i, n_b_i, desired_balance=0.5, max_upsampling=0.0)
        assert np.isclose(n_a_f, 2), "a should not change for pure upsampling"
        assert np.isclose(n_b_f, 2), "did not downsample to 2"

    def test_triple_upsample(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils.get_balance_data_sizes(n_a_i, n_b_i, desired_balance=0.5, max_upsampling=2.0)
        assert np.isclose(n_a_f, 6), "a should be upsampled to 6"
        assert np.isclose(n_b_f, 6), "b should be downsampled to 6"

    def test_not_perfect_balance(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils.get_balance_data_sizes(n_a_i, n_b_i, desired_balance=0.25, max_upsampling=0.0)
        assert np.isclose(n_a_f, 2), "a not change"
        assert np.isclose(n_b_f, 6), "b should be downsampled to meet the balance"

    def test_flipped_major_class(self):
        n_a_i = 10
        n_b_i = 2
        n_a_f, n_b_f =  l2tml_utils.data_utils.get_balance_data_sizes(n_a_i, n_b_i, desired_balance=0.5, max_upsampling=0.0)
        assert np.isclose(n_b_f, 2), "Did not downsample to 2"
        assert np.isclose(n_a_f, 2), "a should not change since it is minority and no upsampling"
