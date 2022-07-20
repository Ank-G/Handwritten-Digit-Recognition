
from context import nnet
from nnet import activation

import unittest
import torch
import math
import random


class TestActivationModule(unittest.TestCase):
    
    def test_sigmoid(self):
        x = torch.FloatTensor([[-140, -0.2, -0.6, 0, 0.1, 0.5, 2, 50], [-1, -20, -0.8, 10, 1, 0.5, 2.771, 41]])
        y = torch.FloatTensor([[4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999], 
                               [4.53979e-05, 0.45016, 0.35434, 0.5, 0.52498, 0.62246, 0.88079, 0.9999]])
        z = torch.FloatTensor([[2.3459e-05, -0.45016, -0.75234, 0.5984, 0.52498, 0.62246e-05, 0.90079, 0.9219], 
                               [4.53979, 0.45016, 0.35434, 0.9872, 0.54328, 0.62246, 0.88079, 0.7729]])
        a = torch.FloatTensor([[-8, -0.2e-05, -0.845, 0.974, 0.01, 0.45365, 2.2127, -50], [-1, 20, -0.887, -10, 5.2425, 0.5, -4.771, 0]])
        b = torch.FloatTensor([[-2.3459e-05, -0.45016, -0.75234, -0.5984, -0.498, -0.23246, -0.2379, -0.1099], 
                               [-4.53979, -0.42346, -0.98735, -0.7772, -0.3452328, -0.62246, -0.88079, -0.7729]])
        c = torch.FloatTensor([[30, 0.9, 0.23, 0.43, 9.1, 0.5, 8.5, 50], [1, 10.345, 0.835, 10, 14, 0.5, 7.771, 21]])   

        precision = 0.000001

        self.assertTrue(torch.le(torch.abs(activation.sigmoid(x) - x.sigmoid()), precision).all())
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(y) - y.sigmoid()), precision).all())
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(z) - z.sigmoid()), precision).all())
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(a) - a.sigmoid()), precision).all())
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(b) - b.sigmoid()), precision).all())
        self.assertTrue(torch.le(torch.abs(activation.sigmoid(c) - c.sigmoid()), precision).all())

    def test_delta_sigmoid(self):
        batch_size = 4
        N_hn = 256
        precision = 0.000001 

        x = torch.rand((batch_size, N_hn), dtype=torch.float, requires_grad=True)
        y = torch.full((batch_size, N_hn), 0.5, dtype=torch.float, requires_grad=True) # all 0.5
        z = torch.rand((batch_size, N_hn), dtype=torch.float, requires_grad=True)
        a = torch.zeros((batch_size, N_hn), dtype=torch.float, requires_grad=True) # all zeros
        b = torch.rand((batch_size, N_hn), dtype=torch.float, requires_grad=True)
        c = torch.rand((batch_size, N_hn), dtype=torch.float, requires_grad=True)

        grads_x = activation.delta_sigmoid(x)
        grads_y = activation.delta_sigmoid(y)
        grads_z = activation.delta_sigmoid(z)
        grads_a = activation.delta_sigmoid(a)
        grads_b = activation.delta_sigmoid(b)
        grads_c = activation.delta_sigmoid(c)
        
        # calculate gradients with torch
        x.sigmoid().backward(torch.ones_like(x))
        y.sigmoid().backward(torch.ones_like(y))
        z.sigmoid().backward(torch.ones_like(z))
        a.sigmoid().backward(torch.ones_like(a))
        b.sigmoid().backward(torch.ones_like(b))
        c.sigmoid().backward(torch.ones_like(c))
        
        assert isinstance(grads_x, torch.FloatTensor)
        assert grads_x.size() == torch.Size([batch_size, N_hn])
        self.assertTrue(torch.le(torch.abs(grads_x - x.grad), precision).all())
        self.assertTrue(torch.le(torch.abs(grads_y - y.grad), precision).all())
        self.assertTrue(torch.le(torch.abs(grads_z - z.grad), precision).all())
        self.assertTrue(torch.le(torch.abs(grads_a - a.grad), precision).all())
        self.assertTrue(torch.le(torch.abs(grads_b - b.grad), precision).all())
        self.assertTrue(torch.le(torch.abs(grads_c - c.grad), precision).all())

    def test_softmax(self):
        batch_size = 4
        N_out = 10
        precision = 0.000001
        
        x = torch.rand((batch_size, N_out), dtype=torch.float)
        y = torch.zeros((batch_size, N_out), dtype=torch.float) # all zeros
        z = torch.rand((batch_size, N_out), dtype=torch.float)
        a = torch.full((batch_size, N_out), 0.6784, dtype=torch.float) # all 0.6784
        b = torch.rand((batch_size, N_out), dtype=torch.float)
        c = torch.rand((batch_size, N_out), dtype=torch.float)

        outputs_x = activation.softmax(x)
        outputs_y = activation.softmax(y)
        outputs_z = activation.softmax(z)
        outputs_a = activation.softmax(a)
        outputs_b = activation.softmax(b)
        outputs_c = activation.softmax(c)
        
        assert isinstance(outputs_x, torch.FloatTensor)
        assert outputs_x.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(outputs_x - x.softmax(1)), precision).all())  
        self.assertTrue(torch.le(torch.abs(outputs_y - y.softmax(1)), precision).all())
        self.assertTrue(torch.le(torch.abs(outputs_z - z.softmax(1)), precision).all())
        self.assertTrue(torch.le(torch.abs(outputs_a - a.softmax(1)), precision).all())
        self.assertTrue(torch.le(torch.abs(outputs_b - b.softmax(1)), precision).all())
        self.assertTrue(torch.le(torch.abs(outputs_c - c.softmax(1)), precision).all())


if __name__ == '__main__':
    unittest.main()