
from context import nnet
from nnet import loss, activation

import unittest
import torch
import math
import numpy as np


class TestLossModule(unittest.TestCase):

    def test_cross_entropy(self):
        # settings
        batch_size = 4
        N_out = 10

        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float)
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        creloss = loss.cross_entropy_loss(activation.softmax(outputs), labels)
        assert type(creloss) == float

        nll = torch.nn.functional.cross_entropy(outputs, labels)
        self.assertAlmostEqual(creloss, nll.item(), places=6)
        ####################################################################################################################
        outputs = torch.zeros((batch_size, N_out), dtype=torch.float) # all zeros
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        creloss = loss.cross_entropy_loss(activation.softmax(outputs), labels)

        nll = torch.nn.functional.cross_entropy(outputs, labels)
        self.assertAlmostEqual(creloss, nll.item(), places=6)
        ####################################################################################################################
        outputs = torch.full((batch_size, N_out), 0.5, dtype=torch.float) # all 0.5
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        creloss = loss.cross_entropy_loss(activation.softmax(outputs), labels)

        nll = torch.nn.functional.cross_entropy(outputs, labels)
        self.assertAlmostEqual(creloss, nll.item(), places=6)


    def test_delta_cross_entropy_loss(self):
        # settings
        batch_size = 4
        N_out = 10
        precision = 0.000001

        # tensors
        outputs = torch.rand((batch_size, N_out), dtype=torch.float, requires_grad=True)
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        # calculate gradients from scratch
        grads_creloss = loss.delta_cross_entropy_softmax(activation.softmax(outputs), labels)
        
        # calculate gradients with autograd
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        nll.backward()

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())
        ####################################################################################################################
        outputs = torch.zeros((batch_size, N_out), dtype=torch.float, requires_grad=True) # all zeros
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        grads_creloss = loss.delta_cross_entropy_softmax(activation.softmax(outputs), labels)
        
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        nll.backward()

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())
        ####################################################################################################################
        outputs = torch.full((batch_size, N_out), 0.7354, dtype=torch.float, requires_grad=True) # all 0.7354
        labels = torch.randint(high=N_out, size=(batch_size,), dtype=torch.long)

        grads_creloss = loss.delta_cross_entropy_softmax(activation.softmax(outputs), labels)
        
        nll = torch.nn.functional.cross_entropy(outputs, labels)
        nll.backward()

        assert isinstance(grads_creloss, torch.FloatTensor)
        assert grads_creloss.size() == torch.Size([batch_size, N_out])
        self.assertTrue(torch.le(torch.abs(grads_creloss - outputs.grad), precision).all())


if __name__ == '__main__':
    unittest.main()
