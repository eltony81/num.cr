# This example demonstrates implementing classic Machine Learning algorithms
# using num.cr.
#
# It covers:
# 1. Analytical Linear Regression using standard normal equations (Least Squares).
# 2. Iterative Linear Regression using Autograd (`Num::Grad::Context`).
# 3. Logistic Regression for binary classification.
#
# To run this example:
#   crystal run examples/machine_learning_usecases.cr

require "../src/num"

Num::Rand.set_seed(42)

# =====================================================================
# 1. Analytical Linear Regression (Least Squares)
# =====================================================================
puts "=== 1. Analytical Linear Regression (Least Squares) ==="
# Goal: Fit y = w1*x1 + w2*x2 + b
# Mathematically, we add a column of 1s to X for the bias term (b),
# and solve the Normal Equations: W_opt = (X^T * X)^-1 * X^T * y

# Generate synthetic dataset: 100 samples, 2 features
n_samples = 100
n_features = 2

# X shape: [100, 2]
x_data = Tensor(Float64, CPU(Float64)).normal([n_samples, n_features], loc: 0.0, sigma: 1.0)

# True parameters: weights = [2.5, -1.5], bias = 4.2
true_w = [[2.5], [-1.5]].to_tensor
true_b = 4.2

# y = X * w + b + noise
noise = Tensor(Float64, CPU(Float64)).normal([n_samples], loc: 0.0, sigma: 0.1)
y_data = (x_data.matmul(true_w) + true_b).reshape([n_samples]) + noise

# Add bias column (all ones) to X to get design matrix X_design of shape [100, 3]
ones = Tensor(Float64, CPU(Float64)).ones([n_samples, 1])
x_design = Num.concatenate([ones, x_data], axis: 1) # Concatenate along axis 1

# Normal Equation: (X^T * X) * W = X^T * y
# Let A = X^T * X, B = X^T * y. Solve A * W = B
xt = x_design.transpose
a = xt.matmul(x_design)
b = xt.matmul(y_data.reshape([n_samples, 1]))

# Solve system
w_opt = a.solve(b)

puts "True weights: #{true_w.to_a}, True bias: #{true_b}"
puts "Estimated bias (intercept): #{w_opt[0]}"
puts "Estimated weights: [#{w_opt[1]}, #{w_opt[2]}]"
puts "\n"


# =====================================================================
# 2. Linear Regression via Gradient Descent (Autograd)
# =====================================================================
puts "=== 2. Linear Regression via Autograd Gradient Descent ==="

# Initialize computation context
ctx = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new

# Variables to optimize (requires_grad: true)
w_var = ctx.variable(Tensor(Float64, CPU(Float64)).normal([2, 1], loc: 0.0, sigma: 1.0), requires_grad: true)
b_var = ctx.variable(Tensor(Float64, CPU(Float64)).zeros([n_samples, 1]), requires_grad: true)

# Inputs (do not require gradients)
x_var = ctx.variable(x_data, requires_grad: false)
y_var = ctx.variable(y_data.reshape([n_samples, 1]), requires_grad: false)

# Learning rate and epochs
learning_rate = 0.1
epochs = 50

epochs.times do |epoch|
  # Forward pass: y_pred = X * W + b
  y_pred = x_var.matmul(w_var) + b_var

  # Compute Mean Squared Error Loss: MSE = mean((y_pred - y)**2)
  error = y_pred - y_var
  loss = (error * error).mean(0)

  # Backpropagation
  loss.backprop

  # Update parameters manually using SGD: param = param - lr * grad
  w_var.value.map!(w_var.grad) { |w, g| w - learning_rate * g }
  b_var.value.map!(b_var.grad) { |bias, g| bias - learning_rate * g }

  # Reset gradients to zero for the next iteration
  w_var.grad.map! { 0.0 }
  b_var.grad.map! { 0.0 }

  if (epoch + 1) % 10 == 0
    puts "Epoch #{epoch + 1} / #{epochs} | Loss: #{loss.value.value}"
  end
end

puts "Estimated weights (GD): #{w_var.value[0...2, 0].to_a}"
puts "Estimated bias (GD): #{b_var.value[0, 0]}"
puts "\n"


# =====================================================================
# 3. Logistic Regression for Binary Classification (Autograd)
# =====================================================================
puts "=== 3. Logistic Regression (Binary Classification) ==="

# Generate simple classification dataset: two clusters
n_cls_samples = 80
# Cluster 1 centered at [1.5, 1.5] -> label 1
class1_x = Tensor(Float64, CPU(Float64)).normal([n_cls_samples // 2, 2], loc: 1.5, sigma: 0.5)
class1_y = Tensor(Float64, CPU(Float64)).ones([n_cls_samples // 2, 1])

# Cluster 2 centered at [-1.5, -1.5] -> label 0
class0_x = Tensor(Float64, CPU(Float64)).normal([n_cls_samples // 2, 2], loc: -1.5, sigma: 0.5)
class0_y = Tensor(Float64, CPU(Float64)).zeros([n_cls_samples // 2, 1])

# Combine clusters
x_cls = Num.concatenate([class1_x, class0_x], axis: 0)
y_cls = Num.concatenate([class1_y, class0_y], axis: 0)

# Initialize variables
ctx_logistic = Num::Grad::Context(Tensor(Float64, CPU(Float64))).new
w_cls = ctx_logistic.variable(Tensor(Float64, CPU(Float64)).normal([2, 1], loc: 0.0, sigma: 1.0), requires_grad: true)
b_cls = ctx_logistic.variable(Tensor(Float64, CPU(Float64)).zeros([n_cls_samples, 1]), requires_grad: true)

x_cls_var = ctx_logistic.variable(x_cls, requires_grad: false)
y_cls_var = ctx_logistic.variable(y_cls, requires_grad: false)

# Wrap constant variables for mathematical operations in the graph
one_var = ctx_logistic.variable(Tensor(Float64, CPU(Float64)).ones([1]), requires_grad: false)
eps_var = ctx_logistic.variable(Tensor(Float64, CPU(Float64)).ones([1]) * 1e-15, requires_grad: false)

lr_cls = 0.2
epochs_cls = 100

epochs_cls.times do |epoch|
  # Forward: logits = X * W + b
  logits = x_cls_var.matmul(w_cls) + b_cls

  # Sigmoid function: y_prob = 1 / (1 + exp(-logits))
  # Using autograd operators:
  y_prob = one_var / (one_var + (-logits).exp)

  # Binary Cross Entropy Loss: Loss = -mean(y * log(prob) + (1 - y) * log(1 - prob))
  loss_item = -((y_cls_var * (y_prob + eps_var).log) + ((one_var - y_cls_var) * (one_var - y_prob + eps_var).log))
  loss = loss_item.mean(0)

  loss.backprop

  # Update parameters
  w_cls.value.map!(w_cls.grad) { |w, g| w - lr_cls * g }
  b_cls.value.map!(b_cls.grad) { |b, g| b - lr_cls * g }

  # Zero grads
  w_cls.grad.map! { 0.0 }
  b_cls.grad.map! { 0.0 }

  if (epoch + 1) % 20 == 0
    # Calculate training accuracy
    predictions = y_prob.value.map { |p| p >= 0.5 ? 1.0 : 0.0 }
    correct = 0
    predictions.each_with_index do |p, i|
      correct += 1 if p == y_cls[i, 0].value
    end
    accuracy = correct.to_f / n_cls_samples
    puts "Epoch #{epoch + 1} / #{epochs_cls} | Loss: #{loss.value.value.round(4)} | Accuracy: #{(accuracy * 100).round(2)}%"
  end
end

puts "Final Weights:\n#{w_cls.value}"
puts "Final Bias: #{b_cls.value.to_a[0]}"
