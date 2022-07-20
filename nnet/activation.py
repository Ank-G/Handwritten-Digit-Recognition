
import torch


def sigmoid(z):
    """Calculates sigmoid values for tensors

    This function is used to calculate the sigmoid function for the given inputs.

    Args:
        z (torch.tensor): the inputs to the hidden layer. Size (batch_size, N_h1) 

    Returns:
        result (torch.tensor): the output of the sigmoid function. Size like z 
    """
    result = torch.div(torch.ones_like(z), torch.add(torch.ones_like(z), 1.0, torch.exp(-z)))

    return result

def delta_sigmoid(z):
    """Calculates derivative of sigmoid function

    This function is used to calculate the derivative of sigmoid function for the given inputs - used in backpropagation

    Args:
        z (torch.tensor): the inputs to the hidden layer. Size (batch_size, N_h1) 

    Returns:
        grad_sigmoid (torch.tensor): the derivative of the sigmoid function. Size like z   
    """
    z = sigmoid(z)
    grad_sigmoid = z * (torch.ones_like(z) - z)
    
    return grad_sigmoid

def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

    This function is used to calculate the softmax function for the given inputs.

    Args:
        x (torch.tensor): the inputs to the output layer. Size (batch_size, N_out) 

    Returns:
        stable_softmax (torch.tensor): the output of the softmax function. Size like x    
    """
    stable_softmax = torch.empty(x.shape)
    for i in range(x.shape[0]):
        z = x - max(x[i]) # to prevent exponential from overflow and underflow
        stable_softmax[i,:] = torch.exp(z[i,:])
        stable_softmax[i,:] /= torch.sum(stable_softmax[i,:])
    
    return stable_softmax


if __name__ == "__main__":
    pass
