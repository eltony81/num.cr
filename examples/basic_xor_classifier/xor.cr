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

# This example demonstrates training a small neural network to learn
# the XOR logical function using the Autograd and NN subsystems in num.cr.

require "../../src/num"

# Set a fixed seed to ensure weights are initialized identically on every run
Num::Rand.set_seed(2)

# Set up the automatic differentiation context to build the computational graph
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

batch_size = 32

# Generate a synthetic dataset of boolean pairs (0 or 1) representing inputs
# Shape: [3200, 2]
x_train_bool = Tensor(Float64, CPU(Float64)).random(0_u8...2_u8, [batch_size * 100, 2])

# Ground truth XOR logic: y = x1 XOR x2
# Slices: x_train_bool[..., ...1] is the first column, x_train_bool[..., 1...] is the second.
# Using bitwise XOR operator '^' element-wise.
y_bool = x_train_bool[..., ...1] ^ x_train_bool[..., 1...]

# Convert raw data tensors to variables in the autograd context
x_train = ctx.variable(x_train_bool.as_type(Float64))
y = y_bool.as_type(Float64)

# Create a Multi-Layer Perceptron neural network
# It contains one hidden layer with 3 neurons and a single output neuron.
net = Num::NN::Network.new(ctx) do
  # Input layer expects a 2-dimensional feature vector
  input [2]
  
  # Hidden layer with 3 units
  linear 3
  
  # Activation function
  relu
  
  # Output layer mapping to 1 final logit
  linear 1
  
  # SGD Optimizer with a learning rate of 0.7
  sgd 0.7
  
  # Loss function suited for binary predictions
  sigmoid_cross_entropy_loss
end

losses = [] of Float64

50.times do |epoch|
  # Loop through our 100 mini-batches
  100.times do |batch_id|
    offset = batch_id * batch_size
    
    # Slice the input and output targets for the current batch
    x = x_train[offset...offset + batch_size]
    target = y[offset...offset + batch_size]

    # 1. Forward pass: compute network output
    y_pred = net.forward(x)

    # 2. Compute Loss
    loss = net.loss(y_pred, target)

    puts "Epoch is: #{epoch}"
    puts "Batch id: #{batch_id}"
    puts "Loss is: #{loss.value.value}"
    losses << loss.value.value

    # 3. Backpropagation: compute gradients of parameters
    loss.backprop
    
    # 4. Update Network weights
    net.optimizer.update
  end
end
