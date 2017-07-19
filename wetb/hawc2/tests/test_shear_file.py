'''
Created on 05/11/2015

@author: MMPE
'''
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from io import open
from future import standard_library
standard_library.install_aliases()
import unittest
from wetb.hawc2 import shear_file
import numpy as np
import os
import shutil
testfilepath = 'test_files/'
class TestShearFile(unittest.TestCase):


    def test_shearfile(self):
        f = testfilepath + "tmp_shearfile1.dat"
        shear_file.save(f, [-55, 55], [30, 100, 160] , u=np.array([[0.7, 1, 1.3], [0.7, 1, 1.3]]).T)
        with open(f) as fid:
            self.assertEqual(fid.read(),
""" # autogenerated shear file
  2 3
 # shear v component
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
 # shear u component
  0.7000000000 0.7000000000
  1.0000000000 1.0000000000
  1.3000000000 1.3000000000
 # shear w component
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
 # v coordinates
  -55.0000000000
  55.0000000000
 # w coordinates
  30.0000000000
  100.0000000000
  160.0000000000
""")
        os.remove(f)


    def test_shearfile2(self):
        f = testfilepath + "tmp_shearfile2.dat"
        shear_file.save(f, [-55, 55], [30, 100, 160] , u=np.array([0.7, 1, 1.3]).T)
        with open(f) as fid:
            self.assertEqual(fid.read(),
""" # autogenerated shear file
  2 3
 # shear v component
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
 # shear u component
  0.7000000000 0.7000000000
  1.0000000000 1.0000000000
  1.3000000000 1.3000000000
 # shear w component
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
  0.0000000000 0.0000000000
 # v coordinates
  -55.0000000000
  55.0000000000
 # w coordinates
  30.0000000000
  100.0000000000
  160.0000000000
""")
        os.remove(f)

    def test_shear_makedirs(self):
        f = testfilepath + "shear/tmp_shearfile2.dat"
        shear_file.save(f, [-55, 55], [30, 100, 160] , u=np.array([0.7, 1, 1.3]).T)
        shutil.rmtree(testfilepath + "shear")

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_shearfile']
    unittest.main()
