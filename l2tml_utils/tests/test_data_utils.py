import pytest
import numpy as np
import l2tml_utils.data_utils



class TestBalanceFn:

    def test_all_upsample(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils(n_a_i, n_b_i, desired_balance=0.5, upsample_frac=1.0)
        assert np.isclose(n_a_f, 10), "Did not upsample to 10"
        assert np.isclose(n_b_f, 10), "b should not change for pure upsampling"

    def test_all_downsample(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils(n_a_i, n_b_i, desired_balance=0.5, upsample_frac=0.0)
        assert np.isclose(n_a_f, 2), "a should not change for pure upsampling"
        assert np.isclose(n_b_f, 2), "did not downsample to 2"

    def test_half_up_half_down(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils(n_a_i, n_b_i, desired_balance=0.5, upsample_frac=0.5)
        assert np.isclose(n_a_f, 6), "a should be upsampled to 6"
        assert np.isclose(n_b_f, 6), "b should be downsampled to 6"

    def test_not_perfect_balance(self):
        n_a_i = 2
        n_b_i = 10
        n_a_f, n_b_f =  l2tml_utils.data_utils(n_a_i, n_b_i, desired_balance=0.3333, upsample_frac=1.0)
        assert np.isclose(n_a_f, 6), "a should be upsampled to 5"
        assert np.isclose(n_b_f, 10), "b should not change for pure upsampling"

    def test_flipped_major_class(self):
        n_a_i = 10
        n_b_i = 2
        n_a_f, n_b_f =  l2tml_utils.data_utils(n_a_i, n_b_i, desired_balance=0.3333, upsample_frac=1.0)
        assert np.isclose(n_b_f, 10), "Did not upsample to 10"
        assert np.isclose(n_a_f, 10), "a should not change for pure upsampling"
