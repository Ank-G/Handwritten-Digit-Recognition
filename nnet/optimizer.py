
import torch


def mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr):
    """Mini-batch gradient descent
    
    This function is used to perform gradient descent to find the minimum of a curve. It is done by updating the weights and biases

    Args:
        weights (torch.tensor): list of current weights. 
        biases (torch.tensor): list of current biases. 
        dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
        db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
        dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
        db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
        dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
        db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        lr (float): the learning rate of the neural network

    Returns:
        weights (torch.tensor): list of updated weights. 
        biases (torch.tensor): list of updated biases. 
    """
    weights['w1'] = torch.add(weights['w1'], -1, (lr*dw1))
    weights['w2'] = torch.add(weights['w2'], -1, (lr*dw2))
    weights['w3'] = torch.add(weights['w3'], -1, (lr*dw3))

    biases['b1'] = torch.add(biases['b1'], -1, (lr*db1))
    biases['b2'] = torch.add(biases['b2'], -1, (lr*db2))
    biases['b3'] = torch.add(biases['b3'], -1, (lr*db3))

    return weights, biases


if __name__ == "__main__":
    pass