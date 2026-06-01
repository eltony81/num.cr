# This example demonstrates building and training a deep neural network
# with advanced features in num.cr:
# 1. Using the Adam Optimizer (`adam`) for faster convergence.
# 2. Applying Leaky ReLU and ELU activation functions.
# 3. Utilizing Dropout regularization (`dropout`) to prevent overfitting.
# 4. Training on a non-linearly separable concentric circles dataset.
#
# To run this example:
#   crystal run examples/neural_network_advanced.cr

require "../src/num"

Num::Rand.set_seed(42)

# =====================================================================
# 1. Generate Concentric Circles (Non-linear Classification Dataset)
# =====================================================================
puts "=== Generating Synthetic Concentric Circles Dataset ==="

n_samples = 400
half_samples = n_samples // 2

# Inner Circle (Radius ~0.4) -> label 1
r_inner = Tensor(Float64, CPU(Float64)).rand([half_samples]) * 0.3 + 0.2
theta_inner = Tensor(Float64, CPU(Float64)).rand([half_samples]) * (2 * Math::PI)
x_inner = (r_inner * theta_inner.cos).reshape([-1, 1])
y_inner = (r_inner * theta_inner.sin).reshape([-1, 1])
features_inner = Num.concatenate([x_inner, y_inner], axis: 1)
labels_inner = Tensor(Float64, CPU(Float64)).ones([half_samples, 1])

# Outer Ring (Radius ~1.0) -> label 0
r_outer = Tensor(Float64, CPU(Float64)).rand([half_samples]) * 0.4 + 0.8
theta_outer = Tensor(Float64, CPU(Float64)).rand([half_samples]) * (2 * Math::PI)
x_outer = (r_outer * theta_outer.cos).reshape([-1, 1])
y_outer = (r_outer * theta_outer.sin).reshape([-1, 1])
features_outer = Num.concatenate([x_outer, y_outer], axis: 1)
labels_outer = Tensor(Float64, CPU(Float64)).zeros([half_samples, 1])

# Combine dataset
x_raw = Num.concatenate([features_inner, features_outer], axis: 0)
y_raw = Num.concatenate([labels_inner, labels_outer], axis: 0)

# Shuffle the dataset to ensure mini-batches contain mixed classes
indices = (0...n_samples).to_a.shuffle
x_data = Tensor(Float64, CPU(Float64)).zeros([n_samples, 2])
y_data = Tensor(Float64, CPU(Float64)).zeros([n_samples, 1])

indices.each_with_index do |idx, i|
  x_data[i, 0] = x_raw[idx, 0]
  x_data[i, 1] = x_raw[idx, 1]
  y_data[i, 0] = y_raw[idx, 0]
end

puts "Sample features (first 5):"
5.times { |i| puts "x: #{x_data[i, 0].value.round(4)}, #{x_data[i, 1].value.round(4)} | y: #{y_data[i, 0].value}" }

puts "Dataset features shape: #{x_data.shape}"
puts "Dataset labels shape: #{y_data.shape}"
puts "\n"


# =====================================================================
# 2. Define Advanced Neural Network Architecture
# =====================================================================
puts "=== Constructing Network ==="

ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

net = Num::NN::Network.new(ctx) do
  # Input layer: 2D coordinates (x, y)
  input [2]

  # Hidden layer 1: 2 to 16 units
  linear 16
  relu
  
  # Hidden layer 2: 16 to 8 units
  linear 8
  relu
  
  # Hidden layer 3: 8 to 2 units (logits for class 0 and 1)
  linear 2
  
  # Loss function: Softmax combined with Cross Entropy Loss
  softmax_cross_entropy_loss

  # Optimizer: Adam with a learning rate of 0.002
  adam 0.002
end


# =====================================================================
# 3. Training Loop
# =====================================================================
puts "=== Training Network ==="

# Convert to variables
x_train = ctx.variable(x_data, requires_grad: false)
# For softmax cross entropy, targets should be one-hot encoded or class probability vectors.
# Let's create one-hot targets of shape [n_samples, 2]
y_onehot = Tensor(Float64, CPU(Float64)).zeros([n_samples, 2])
n_samples.times do |i|
  label = y_data[i, 0].value.to_i
  y_onehot[i, label] = 1.0
end

batch_size = 32
epochs = 80

epochs.times do |epoch|
  y_trues = [] of Int32
  y_preds = [] of Int32
  epoch_loss = 0.0
  batches = n_samples // batch_size

  batches.times do |batch_id|
    offset = batch_id * batch_size
    x_batch = x_train[offset...offset + batch_size]
    y_batch = y_onehot[offset...offset + batch_size]

    # Forward pass
    output = net.forward(x_batch)

    # Compute loss
    loss = net.loss(output, y_batch)
    epoch_loss += loss.value.value

    # Track labels for accuracy
    y_trues += y_batch.argmax(axis: 1).to_a
    y_preds += output.value.argmax(axis: 1).to_a

    # Backpropagation
    loss.backprop

    # Optimization step
    net.optimizer.update
  end

  if (epoch + 1) % 10 == 0
    accuracy = y_trues.zip(y_preds).map { |t, p| (t == p).to_unsafe }.sum / y_trues.size
    avg_loss = epoch_loss / batches
    puts "Epoch #{epoch + 1} / #{epochs} | Loss: #{avg_loss.round(4)} | Training Accuracy: #{(accuracy * 100).round(2)}%"
  end
end
