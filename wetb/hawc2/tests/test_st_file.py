'''
Created on 05/11/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
import os

from numpy import testing

from wetb.hawc2.st_file import StFile


testfilepath = os.path.join(os.path.dirname(__file__), 'test_files/')  # test file path
class TestStFile(unittest.TestCase):


    def test_stfile(self):
        st = StFile(testfilepath + 'DTU_10MW_RWT_Blade_st.dat')
        self.assertEqual(st.radius_st()[2], 3.74238)
        self.assertEqual(st.radius_st(3), 3.74238)
        self.assertEqual(st.x_e(67.7351), 4.4320990737400E-01)
        self.assertEqual(st.E(3.74238, 1, 1), 1.2511695058500E+10)
        self.assertEqual(st.E(3.74238, 1, 2), 1.2511695058500E+27)


    def test_stfile_interpolate(self):
        st = StFile(testfilepath + 'DTU_10MW_RWT_Blade_st.dat')
        self.assertAlmostEqual(st.x_e(72.2261), 0.381148048)
        self.assertAlmostEqual(st.y_e(72.2261), 0.016692967)

    def test_save(self):
        fname = os.path.join(testfilepath, 'DTU_10MW_RWT_Blade_st.dat')
        fname2 = os.path.join(testfilepath, 'DTU_10MW_RWT_Blade_st2.dat')
        st = StFile(fname)
        st.save(fname2, encoding='utf-8', precision='%20.12e')
        st2 = StFile(fname2)
        self.assertEqual(len(st.main_data_sets), len(st2.main_data_sets))
        self.assertEqual(len(st.main_data_sets[1]), len(st2.main_data_sets[1]))
        for k in st.main_data_sets[1]:
            testing.assert_almost_equal(st.main_data_sets[1][k],
                                        st2.main_data_sets[1][k], decimal=12)
        os.remove(fname2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
