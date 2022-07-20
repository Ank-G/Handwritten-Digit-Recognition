
from context import nnet
from nnet import optimizer

import unittest
import torch
import math
import numpy as np


class TestOptimizerModule(unittest.TestCase):
    
    def test_mbgd(self):
        N_in = 28 * 28
        N_h1 = 256
        N_h2 = 256
        N_out = 10
        precision = 0.009
        lr = 0.001

        w1 = torch.randn((N_h1, N_in), dtype=torch.float)
        w2 = torch.randn((N_h2, N_h1), dtype=torch.float)
        w3 = torch.randn((N_out, N_h2), dtype=torch.float)
        b1 = torch.randn(N_h1, dtype=torch.float)
        b2 = torch.randn(N_h2, dtype=torch.float)
        b3 = torch.randn(N_out, dtype=torch.float)
        
        weights = {'w1': w1, 'w2': w2, 'w3': w3}
        biases =  {'b1': b1, 'b2': b2, 'b3': b3}
        all_params = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3}

        dw1 = torch.randn_like(w1)
        db1 = torch.randn_like(b1)
        dw2 = torch.randn_like(w2)
        db2 = torch.randn_like(b2)
        dw3 = torch.randn_like(w3)
        db3 = torch.randn_like(b3)
        
        n_weights, n_biases = optimizer.mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr)
        assert isinstance(n_weights, dict)
        assert isinstance(n_biases, dict)
        
        for params in [n_weights, n_biases]:
            for key, value in params.items():
                assert isinstance(value, torch.FloatTensor)
                assert params[key].size() == all_params[key].size()

        self.assertTrue(torch.le(torch.abs(n_weights['w1'] - weights['w1'] + lr * dw1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_weights['w2'] - weights['w2'] + lr * dw2), precision).all())        
        self.assertTrue(torch.le(torch.abs(n_weights['w3'] - weights['w3'] + lr * dw3), precision).all())
        
        self.assertTrue(torch.le(torch.abs(n_biases['b1'] - biases['b1'] + lr * db1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b2'] - biases['b2'] + lr * db2), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b3'] - biases['b3'] + lr * db3), precision).all())
        ####################################################################################################################
        weights['w1'] = torch.randn((N_h1, N_in), dtype=torch.float)
        weights['w2'] = torch.randn((N_h2, N_h1), dtype=torch.float)
        weights['w3'] = torch.randn((N_out, N_h2), dtype=torch.float)
        biases['b1'] = torch.randn(N_h1, dtype=torch.float)
        biases['b2'] = torch.randn(N_h2, dtype=torch.float)
        biases['b3'] = torch.randn(N_out, dtype=torch.float)

        n_weights, n_biases = optimizer.mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr)
        assert isinstance(n_weights, dict)
        assert isinstance(n_biases, dict)
        
        for params in [n_weights, n_biases]:
            for key, value in params.items():
                assert isinstance(value, torch.FloatTensor)
                assert params[key].size() == all_params[key].size()

        self.assertTrue(torch.le(torch.abs(n_weights['w1'] - weights['w1'] + lr * dw1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_weights['w2'] - weights['w2'] + lr * dw2), precision).all())        
        self.assertTrue(torch.le(torch.abs(n_weights['w3'] - weights['w3'] + lr * dw3), precision).all())
        
        self.assertTrue(torch.le(torch.abs(n_biases['b1'] - biases['b1'] + lr * db1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b2'] - biases['b2'] + lr * db2), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b3'] - biases['b3'] + lr * db3), precision).all())
        ####################################################################################################################
        weights['w1'] = torch.zeros((N_h1, N_in), dtype=torch.float)
        weights['w2'] = torch.zeros((N_h2, N_h1), dtype=torch.float)
        weights['w3'] = torch.zeros((N_out, N_h2), dtype=torch.float)
        biases['b1'] = torch.zeros(N_h1, dtype=torch.float)
        biases['b2'] = torch.zeros(N_h2, dtype=torch.float)
        biases['b3'] = torch.zeros(N_out, dtype=torch.float)

        dw1 = torch.zeros_like(w1)
        db1 = torch.zeros_like(b1)
        dw2 = torch.zeros_like(w2)
        db2 = torch.zeros_like(b2)
        dw3 = torch.zeros_like(w3)
        db3 = torch.zeros_like(b3)

        n_weights, n_biases = optimizer.mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr)
        assert isinstance(n_weights, dict)
        assert isinstance(n_biases, dict)
        
        for params in [n_weights, n_biases]:
            for key, value in params.items():
                assert isinstance(value, torch.FloatTensor)
                assert params[key].size() == all_params[key].size()

        self.assertTrue(torch.le(torch.abs(n_weights['w1'] - weights['w1'] + lr * dw1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_weights['w2'] - weights['w2'] + lr * dw2), precision).all())        
        self.assertTrue(torch.le(torch.abs(n_weights['w3'] - weights['w3'] + lr * dw3), precision).all())
        
        self.assertTrue(torch.le(torch.abs(n_biases['b1'] - biases['b1'] + lr * db1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b2'] - biases['b2'] + lr * db2), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b3'] - biases['b3'] + lr * db3), precision).all())
        ####################################################################################################################
        dw1 = torch.full_like(w1, 0.3546)
        db1 = torch.full_like(b1, 0.2412)
        dw2 = torch.full_like(w2, 0.1359)
        db2 = torch.full_like(b2, 0.0389)
        dw3 = torch.full_like(w3, 0.2461)
        db3 = torch.full_like(b3, 0.6684)

        n_weights, n_biases = optimizer.mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr)
        assert isinstance(n_weights, dict)
        assert isinstance(n_biases, dict)
        
        for params in [n_weights, n_biases]:
            for key, value in params.items():
                assert isinstance(value, torch.FloatTensor)
                assert params[key].size() == all_params[key].size()

        self.assertTrue(torch.le(torch.abs(n_weights['w1'] - weights['w1'] + lr * dw1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_weights['w2'] - weights['w2'] + lr * dw2), precision).all())        
        self.assertTrue(torch.le(torch.abs(n_weights['w3'] - weights['w3'] + lr * dw3), precision).all())
        
        self.assertTrue(torch.le(torch.abs(n_biases['b1'] - biases['b1'] + lr * db1), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b2'] - biases['b2'] + lr * db2), precision).all())
        self.assertTrue(torch.le(torch.abs(n_biases['b3'] - biases['b3'] + lr * db3), precision).all())


if __name__ == '__main__':
    unittest.main()