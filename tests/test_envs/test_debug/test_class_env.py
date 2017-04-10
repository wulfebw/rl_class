
import numpy as np
import os
import sys
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 
    os.pardir)
sys.path.append(os.path.abspath(path))

import rl.envs.debug.class_env as class_env

class TestClassEnv(unittest.TestCase):

    def test_class_env(self):
        min_x, max_x, discount, horizon = 1, 7, 1, 4
        env = class_env.ClassEnv(min_x, max_x, discount, horizon)
        x = env.reset()
        self.assertEqual(x[0], 4)
        self.assertEqual(x[1], 4)
        a = 0
        x, r, done, _ = env.step(a)
        self.assertEqual(x[0], 3)
        self.assertEqual(x[1], 3) 
        env.step(a)
        env.step(a)
        x, r, done, _ = env.step(a)
        self.assertEqual(x[0], 1)
        self.assertEqual(x[1], 0)
        self.assertEqual(r, 1)
        # with horizon 4, after calling step 4 times should be done
        self.assertEqual(done, True)

if __name__ == '__main__':
    unittest.main()