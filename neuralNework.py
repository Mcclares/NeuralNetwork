import numpy as num
from numpy.core.fromnumeric import ndim
from numpy.core.records import array
import scipy.special as spec
class neuralNetwork() :
    def __init__(self, inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = num.random.normal(0.0,pow(self.hnodes, - 0.5),(self.hnodes, self.inodes))
        self.woh = num.random.normal(0.0,pow(self.onodes, - 0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x: spec.expit(x)
        pass
    def train(self, input_list, target_list) :
        inputs = num.array(input_list, ndmin= 2).T
        targets = num.array(target_list,ndmin= 2).T
      
        hidden_inputs = num.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = num.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
       
        output_errors = targets - final_outputs
        hidden_errors = num.dot(self.woh.T , output_errors)
        self.woh +=  self.lr * num.dot(output_errors * final_outputs * (1.0 - final_outputs)), num.transpose(hidden_outputs)
        self.wih +=  self.lr * num.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), num.transpose(inputs)
        pass

    def query(self, inputs_list) :
        inputs = num.array(inputs_list,ndmin= 2).T
      
        hidden_inputs = num.dot(self.wih, inputs )
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = num.dot(self.woh, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3
n = neuralNetwork(input_nodes, hidden_nodes,output_nodes, learning_rate)
n.query([1.0,0.5,-1.5])
data_file = open("mnist_dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()





