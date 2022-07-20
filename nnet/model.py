
import math
import torch
import numpy as np 


class FullyConnected:
    """Constructs the Neural Network architecture.

    Args:
        N_in (int): input size
        N_h1 (int): hidden layer 1 size
        N_h2 (int): hidden layer 2 size
        N_out (int): output size
        device (str, optional): selects device to execute code. Defaults to 'cpu'
    
    Examples:
        >>> network = model.FullyConnected(2000, 512, 256, 5, device='cpu')
        >>> creloss, accuracy, outputs = network.train(inputs, labels)
    """

    def __init__(self, N_in, N_h1, N_h2, N_out, device='cpu'):
        """Initializes weights and biases, and construct neural network architecture.
        
        One [recommended](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) approach is to initialize weights randomly but uniformly in the interval from [-1/n^0.5, 1/n^0.5] where 'n' is number of neurons from incoming layer. For example, number of neurons in incoming layer is 784, then weights should be initialized randomly in uniform interval between [-1/784^0.5, 1/784^0.5].
        
        You should maintain a list of weights and biases which will be initalized here. They should be torch tensors.

        Optionally, you can maintain a list of activations and weighted sum of neurons in a dictionary named Cache to avoid recalculation of those. If tensors are too large it could be an issue.
        """
        self.N_in = N_in
        self.N_h1 = N_h1
        self.N_h2 = N_h2
        self.N_out = N_out

        self.device = torch.device(device)

        w_range = 1/(784**0.5)  
        w1 = torch.empty(N_h1, N_in).uniform_(-w_range, w_range)
        w2 = torch.empty(N_h2, N_h1).uniform_(-w_range, w_range)
        w3 = torch.empty(N_out, N_h2).uniform_(-w_range, w_range)
        self.weights = {'w1': w1, 'w2': w2, 'w3': w3} 

        b1 = torch.ones(N_h1, dtype=torch.float32)
        b2 = torch.ones(N_h2, dtype=torch.float32)
        b3 = torch.ones(N_out, dtype=torch.float32)
        self.biases = {'b1': b1, 'b2': b2, 'b3': b3}

        z1 = torch.zeros(N_h1, 1, dtype=torch.float32)
        z2 = torch.zeros(N_h2, 1, dtype=torch.float32)
        z3 = torch.zeros(N_out, 1, dtype=torch.float32)
        self.cache = {'z1': z1, 'z2': z2, 'z3': z3}

        self.batch_size = 4
        self.accur = 0.0
        self.accur_count = 0.0

    def train(self, inputs, labels, lr=0.001, debug=True):
        """Trains the neural network on given inputs and labels.

        This function will train the neural network on given inputs and minimize the loss by backpropagating and adjusting weights with some optimizer.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            lr (float, optional): learning rate for training. Defaults to 0.001
            debug (bool, optional): prints loss and accuracy on each update. Defaults to False

        Returns:
            creloss (float): average cross entropy loss
            accuracy (float): ratio of correctly classified to total samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        inputs = torch.reshape(inputs, (4, 784)) # converting the 4 * 1 * 28 * 28 inputs tensor to 4 * 784 where 4 is the size of mini batch and 784 is the inputs to 1st hidden layer 
        outputs = self.forward(inputs) # forward pass
        creloss = loss.cross_entropy_loss(outputs, labels) # calculating loss
        accuracy = self.accuracy(outputs, labels) # calculating accuracy

        if debug:
            print('train loss: ', creloss)
            print('train accuracy: ', accuracy)
        
        dw1, db1, dw2, db2, dw3, db3 = self.backward(inputs, labels, outputs)
        self.weights, self.biases = optimizer.mbgd(self.weights, self.biases, dw1, db1, dw2, db2, dw3, db3, lr)
        
        return creloss, accuracy, outputs

    def predict(self, inputs):
        """Predicts output probability and index of most activating neuron

        This function is used to predict output given inputs. You can then use index in classes to show which class got activated. For example, if in case of MNIST fifth neuron has highest firing probability, then class[5] is the label of input.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 

        Returns:
            score (torch.tensor): max score for each class. Size (batch_size)
            idx (torch.tensor): index of most activating neuron. Size (batch_size)  
        """
        inputs = torch.reshape(inputs, (4, 784))

        outputs = self.forward(inputs) # forward pass
        score, idx = torch.max(outputs, dim=1, keepdim=False) # finding max score and its index
        
        return score, idx

    def eval(self, inputs, labels, debug=True):
        """Evaluate performance of neural network on inputs with labels.

        This function is used to evaluate loss and accuracy of neural network on new examples. Unlike predict(), this function will not only predict but also calculate and return loss and accuracy w.r.t given inputs and labels.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            debug (bool, optional): print loss and accuracy on every iteration. Defaults to False

        Returns:
            loss (float): average cross entropy loss
            accuracy (float): ratio of correctly to uncorrectly classified samples
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        inputs = torch.reshape(inputs, (4, 784))
        outputs = self.forward(inputs) # forward pass
        creloss = loss.cross_entropy_loss(outputs, labels) # calculate loss
        accuracy = self.accuracy(outputs, labels) # calculate accuracy

        if debug:
            print('eval loss: ', creloss)
            print('eval accuracy: ', accuracy)
            
        return creloss, accuracy, outputs

    def accuracy(self, outputs, labels):
        """Accuracy of neural network for given outputs and labels.
        
        Calculates ratio of number of correct outputs to total number of examples.

        Args:
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
            labels (torch.tensor): correct labels. Size (batch_size)
        
        Returns:
            accuracy (float): accuracy score 
        """
        out_max = torch.argmax(outputs, dim=1) # taking the indices of the maximum values in each row
        correct = (out_max == labels).float().sum()
    
        accuracy = correct/outputs.shape[0]
        self.accur_count += 1.0
        self.accur = (self.accur + float(accuracy))
        return self.accur/self.accur_count

    def accuracy_reset(self):
        """Accuracy resetter to reset the accuracy

        This function is used to reset the accuracy. Used to reset accuracy to 0 after each epoch of training and evaluation 

        Args:
            None
        
        Returns:
            None
        """
        self.accur = 0.0
        self.accur_count = 0.0

    def forward(self, inputs):
        """Forward pass of neural network

        Calculates score for each class.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 

        Returns:
            outputs (torch.tensor): predictions from neural network. Size (batch_size, N_out)
        """
        self.cache['z1'] = self.weighted_sum(inputs, self.weights['w1'], self.biases['b1'])
        a1 = activation.sigmoid(self.cache['z1'])

        self.cache['z2'] = self.weighted_sum(a1, self.weights['w2'], self.biases['b2'])
        a2 = activation.sigmoid(self.cache['z2'])

        self.cache['z3'] = self.weighted_sum(a2, self.weights['w3'], self.biases['b3'])
        outputs = activation.softmax(self.cache['z3'])

        return outputs

    def weighted_sum(self, X, w, b):
        """Weighted sum at neuron
        
        Args:
            X (torch.tensor): matrix of Size (K, L)
            w (torch.tensor): weight matrix of Size (J, L)
            b (torch.tensor): vector of Size (J)

        Returns:
            result (torch.tensor): w*X + b of Size (K, J)
        """
        result = torch.matmul(X, torch.transpose(w, 0, 1)) + torch.unsqueeze(b, dim=0)

        return result

    def backward(self, inputs, labels, outputs):
        """Backward pass of neural network
        
        Changes weights and biases of each layer to reduce loss
        
        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            labels (torch.tensor): correct labels. Size (batch_size)
            outputs (torch.tensor): outputs predicted by neural network. Size (batch_size, N_out)
        
        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        dout = loss.delta_cross_entropy_softmax(outputs, labels)
        d2 = torch.matmul(dout, self.weights['w3']) * activation.delta_sigmoid(self.cache['z2'])
        d1 = torch.matmul(d2, self.weights['w2']) * activation.delta_sigmoid(self.cache['z1'])

        dw1, db1, dw2, db2, dw3, db3 = self.calculate_grad(inputs, d1, d2, dout) # calculating all gradients

        return dw1, db1, dw2, db2, dw3, db3

    def calculate_grad(self, inputs, d1, d2, dout):
        """Calculates gradients for backpropagation
        
        This function is used to calculate gradients like loss w.r.t. weights and biases.

        Args:
            inputs (torch.tensor): inputs to train neural network. Size (batch_size, N_in) 
            dout (torch.tensor): error at output. Size like aout or a3 (or z3)
            d2 (torch.tensor): error at hidden layer 2. Size like a2 (or z2)
            d1 (torch.tensor): error at hidden layer 1. Size like a1 (or z1)

        Returns:
            dw1 (torch.tensor): Gradient of loss w.r.t. w1. Size like w1
            db1 (torch.tensor): Gradient of loss w.r.t. b1. Size like b1
            dw2 (torch.tensor): Gradient of loss w.r.t. w2. Size like w2
            db2 (torch.tensor): Gradient of loss w.r.t. b2. Size like b2
            dw3 (torch.tensor): Gradient of loss w.r.t. w3. Size like w3
            db3 (torch.tensor): Gradient of loss w.r.t. b3. Size like b3
        """
        dw3 = torch.matmul(torch.transpose(dout, 0, 1), activation.sigmoid(self.cache['z2']))
        dw2 = torch.matmul(torch.transpose(activation.sigmoid(self.cache['z1']), 0, 1), d2)
        dw1 = torch.matmul(torch.transpose(d1, 0, 1), inputs)

        db3 = np.sum(dout.numpy(), axis=0, keepdims=True) 
        db2 = np.sum(d2.numpy(), axis=0, keepdims=True) 
        db1 = np.sum(d1.numpy(), axis=0, keepdims=True) 

        db3 = torch.squeeze(torch.from_numpy(db3))
        db2 = torch.squeeze(torch.from_numpy(db2))
        db1 = torch.squeeze(torch.from_numpy(db1))

        return dw1, db1, dw2, db2, dw3, db3


if __name__ == "__main__":
    import activation, loss, optimizer
else:
    from nnet import activation, loss, optimizer

