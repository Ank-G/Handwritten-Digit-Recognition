
import torch


def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    This function is used to calculate the cross entropy loss in the output layer.

    Args:
        outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        labels (torch.tensor): correct labels. Size (batch_size)

    Returns:
        creloss.item() (float): the average cross entropy loss as a float value
    """    
    hot_encoded = torch.zeros_like(outputs, dtype=torch.float32)
    for i in range(outputs.shape[0]):
        j = labels[i]
        hot_encoded[i][j] = 1.0

    creloss = -(hot_encoded * torch.log(outputs)).sum(1).mean(0)

    return creloss.item() # should return float not tensor

def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    This function is used to find the derivative of cross entropy loss w.r.t. weighted sum of inputs - used in backpropagation.

    Args:
        outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        labels (torch.tensor): correct labels. Size (batch_size)

    Returns:
        avg_grads (torch.tensor): the derivative of cross entropy loss w.r.t. weighted sum of inputs. Size (batch_size, N_out)
    """
    outputs[range(labels.shape[0]), labels.type(torch.LongTensor)] -= 1
    avg_grads = outputs/labels.shape[0]

    return avg_grads


if __name__ == "__main__":
    pass