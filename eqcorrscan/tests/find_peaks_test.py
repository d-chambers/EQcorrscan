"""
Functions for testing the utils.findpeaks functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from os.path import join

import numpy as np
import pytest

from eqcorrscan.utils.findpeaks import coin_trig
from eqcorrscan.utils.findpeaks import find_peaks, find_peaks2_short


class TestStandardPeakFinding:
    """ Run peak finding against a standard cc array """
    trig_index = 10

    # fixtures
    @pytest.fixture
    def cc_array(self):
        """ load the test cc array case """
        return np.load(join(pytest.test_data_path, 'test_ccc.npy'))

    @pytest.fixture
    def expected_peak_array(self):
        """ load the peak array from running find peaks on cc_arary """
        return np.load(join(pytest.test_data_path, 'test_peaks.npy'))

    @pytest.fixture
    def peak_array(self, cc_array):
        """ run find_peaks2_short on cc_array and return results """
        peaks = find_peaks(arr=cc_array, thresh=0.2, trig_int=self.trig_index,
                           debug=0, starttime=None, samp_rate=200.0)
        return peaks

    @pytest.fixture
    def old_peak_array(self, cc_array):
        return find_peaks2_short(cc_array, thresh=0.2,
                                 trig_int=self.trig_index, samp_rate=200)

    # tests
    def test_max_values(self, old_peak_array, cc_array):
        """ ensure the values in the peak array are max values in 
        the expected window """
        abs_array = np.abs(cc_array)
        for val, ind in old_peak_array:
            assert val == cc_array[ind]
            start, stop = ind - self.trig_index, ind + self.trig_index + 1
            window = abs_array[start: stop]
            assert np.max(window) == np.abs(val)

    def test_compare(self, old_peak_array, peak_array, cc_array):
        assert len(old_peak_array) == len(peak_array)
        assert old_peak_array == peak_array

    def test_main_find_peaks(self, peak_array, expected_peak_array):
        """Test find_peaks2_short returns expected peaks """

        # Check length first as this will be a more obvious issue
        assert len(peak_array) == len(expected_peak_array), (
            'Peaks are not the same length, has ccc been updated?')
        assert (np.array(peak_array) == expected_peak_array).all()


class TestCoincidenceTrigger:
    # fixtures
    @pytest.fixture
    def peaks(self):
        """" create a sample peak array """
        peaks = [[(0.5, 100), (0.3, 800), (0.3, 105)],
                 [(0.4, 120), (0.7, 850)]]
        return peaks

    # tests
    def test_coincidence(self, peaks):
        """Test the coincidence trigger."""

        triggers = coin_trig(peaks, [('a', 'Z'), ('b', 'Z')], samp_rate=10,
                             moveout=3, min_trig=2, trig_int=1)
        assert triggers, [(0.45, 100)]
