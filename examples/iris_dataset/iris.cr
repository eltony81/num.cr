# Copyright (c) 2020 Crystal Data Contributors
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# This example demonstrates classifying the Iris flower dataset
# using a Multi-Layer Perceptron (MLP) built with the Neural Network (NN)
# module and Autograd system of num.cr.

require "../../src/num"

# Fix the random seed for reproducible initialization of weights and shuffling
Num::Rand.set_seed(2)

# Create an autograd context. This context tracks mathematical operations
# on its variables to construct a computational graph for backpropagation.
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

# Load the classic Iris flower dataset containing:
# - labels: Target flower class names (setosa, versicolor, virginica)
# - x_train: Input features (sepal length/width, petal length/width)
# - y_train: One-hot encoded target outputs [150, 3]
labels, x_train, y_train = Num::NN.load_iris_dataset

# Normalize features to have mean=0 and variance=1 (standardization).
# Standardized inputs significantly speed up gradient descent convergence.
x_train = (x_train - x_train.mean(axis: 0)) / x_train.std(axis: 0)

# Convert training features into an autograd variable.
# Since we don't optimize the input dataset features, requires_grad is false (default).
x_train = ctx.variable(x_train)

# Define the Multi-Layer Perceptron neural network architecture
net = Num::NN::Network.new(ctx) do
  # 1. Define the input shape: Iris features have a size of 4
  input [4]
  
  # 2. Fully connected layer projecting 4 features to 3 hidden units
  linear 3
  
  # 3. Activation function (Rectified Linear Unit) to introduce non-linearity
  relu
  
  # 4. Final linear layer projecting 3 hidden units to 3 outputs (corresponding to the classes)
  linear 3
  
  # 5. Optimization algorithm: Stochastic Gradient Descent (SGD) with learning rate 0.9
  sgd 0.9
  
  # 6. Loss function combining sigmoid probabilities with cross entropy loss
  sigmoid_cross_entropy_loss
end

# Set hyper-parameters
batch_size = 10
epochs = 10

epochs.times do |epoch|
  y_trues = [] of Int32
  y_preds = [] of Int32

  # Iterate over batches
  (y_train.shape[0] // batch_size).times do |batch_id|
    # Compute slicing offsets
    offset = batch_id * batch_size
    
    # Slice the training batch features and labels
    x = x_train[offset...offset + batch_size]
    target = y_train[offset...offset + batch_size]

    # 1. Forward Pass: compute prediction logits from the network
    output = net.forward(x)

    # 2. Compute Loss: calculate cross-entropy error against the target
    loss = net.loss(output, target)

    # Extract target class index (argmax over class probabilities)
    y_trues += target.argmax(axis: 1).to_a
    # Extract predicted class index
    y_preds += output.value.argmax(axis: 1).to_a

    # 3. Backward Pass: compute gradients of parameters with respect to loss
    loss.backprop
    
    # 4. Optimization step: update weights using the SGD optimizer
    net.optimizer.update
  end

  # Compute accuracy metric for the epoch
  accuracy = y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size
  puts "Epoch: #{epoch} | Accuracy: #{(accuracy * 100).round(2)}%"
end
