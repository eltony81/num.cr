# This example demonstrates training a Convolutional Neural Network (CNN)
# to classify handwritten digits from the MNIST dataset using num.cr.

require "../../src/num"

# Load the MNIST dataset (handwritten digit images, 28x28 pixels)
dataset = Num::NN.load_mnist_dataset

# Create an autograd context targeting 32-bit floats on the CPU
ctx = Num::Grad::Context(Tensor(Float32, CPU(Float32))).new

batch_size = 32

# Define the Convolutional Neural Network architecture
net = Num::NN::Network.new(ctx) do
  # Input layer expects single-channel (grayscale) 28x28 pixel images
  input [1, 28, 28]
  
  # 1. 2D Convolutional layer: 20 output channels, 5x5 kernel size
  conv2d 20, 5, 5
  
  # Activation function
  relu
  
  # 2. Max pooling: 2x2 window size, 0 padding, stride 2
  maxpool({2, 2}, {0, 0}, {2, 2})
  
  # 3. Second 2D Convolutional layer: 20 output channels, 5x5 kernel size
  conv2d 20, 5, 5
  
  # 4. Second Max pooling: 2x2 window size
  maxpool({2, 2}, {0, 0}, {2, 2})
  
  # 5. Flatten the 3D output of convolutional layers into a 1D vector
  flatten
  
  # 6. Linear layer projecting flattened features to 10 hidden units
  linear 10
  
  # Activation function
  relu
  
  # 7. Output layer projecting to 10 class logits (digits 0 to 9)
  linear 10
  
  # Softmax combined with Cross Entropy loss function
  softmax_cross_entropy_loss
  
  # SGD optimizer with a learning rate of 0.01
  sgd 0.01
end

# Preprocess features: normalize values to [0.0, 1.0] by dividing by 255.
# Reshape the flat feature matrix to match the [batch, channel, height, width] CNN shape.
x_train = ctx.variable((dataset.features / 255_f32).reshape(-1, 1, 28, 28))
y_train = dataset.labels

losses = [] of Float32

10.times do |epoch|
  y_trues = [] of Int32
  y_preds = [] of Int32

  # Iterate over mini-batches
  (x_train.value.shape[0] // batch_size).times do |batch_id|
    offset = batch_id * batch_size
    
    # Slice features and target labels
    x = x_train[offset...offset + batch_size]
    target = y_train[offset...offset + batch_size]

    # 1. Forward pass: compute predictions
    output = net.forward(x)

    # 2. Compute Loss
    loss = net.loss(output, target)
    losses << loss.value.value

    # Track classification labels for accuracy computation
    y_trues += target.argmax(axis: 1).to_a
    y_preds += output.value.argmax(axis: 1).to_a

    # 3. Backward pass: compute gradients
    loss.backprop
    
    # 4. Update model parameters
    net.optimizer.update
  end

  # Compute accuracy metric
  accuracy = y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size
  puts "Epoch: #{epoch} | Accuracy: #{(accuracy * 100).round(2)}%"
end
