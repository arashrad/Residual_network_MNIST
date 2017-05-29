from __future__ import print_function

import tensorflow as tf
import numpy as np


# Reading the MNIST data-set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Network's hyper-parameters
learning_rate = 0.001
number_epochs = 20
batch_size = 100
n_input = 784
n_categories = 10
n_hidden_b1 = 256 # number of hidden units for each layer in residual block one.
n_hidden_b2 = 128 # number of hidden units for each layer in residual block two.
n_hidden_b3 = 64 # number of hidden units for each layer in residual block three.

# Model's input and output
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


# defining the building blocks for our residual model
def residual_block(inputs, weights, biases):
    """this function makes a building block for residual network.
    Each block consists of 3 hidden layers with a skip connection between hidden layer 1 and 3.
    Let x be the input of the block and g the activation function (here sigmoid) then the output would be:
    out = h(3)= g(h(2)*W(2) + b(2) + h(1)W(s))...
    """

    # first layer of the block with sigmoid activation function: sigmoid(x.W(1) + b(1))
    hidden_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs, weights['layer1']), biases['layer1']))

    # second layer of the block with sigmoid activation function: sigmoid(hidden_layer_1.W(2) + b(2))
    hidden_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, weights['layer2']), biases['layer2']))

    # third layer of the block: sigmoid(hidden_layer_2.W(3) + b(3) + hidden_layer_1.W(s))
    hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, weights['layer3']), biases['layer3'])
    skip_values = tf.matmul(hidden_layer_1, weights['skip'])
    output = tf.nn.sigmoid(tf.add(hidden_layer_3, skip_values))
    return output


# defining and initiating the weights for blocks. We use Xavier method for weight initiation

low1 = -4*np.sqrt(6.0/(n_input + n_hidden_b1)) # use 4 for sigmoid, 1 for tanh activation
high1 = 4*np.sqrt(6.0/(n_input + n_hidden_b1))
low2 = -4*np.sqrt(6.0/(n_hidden_b1 + n_hidden_b1))
high2 = 4*np.sqrt(6.0/(n_hidden_b1 + n_hidden_b1))


weights_block1 = {
    'layer1': tf.Variable(tf.random_uniform([n_input, n_hidden_b1], minval=low1, maxval=high1, dtype=tf.float32)),
    'layer2': tf.Variable(tf.random_uniform([n_hidden_b1, n_hidden_b1], minval=low2, maxval=high2, dtype=tf.float32)),
    'layer3': tf.Variable(tf.random_uniform([n_hidden_b1, n_hidden_b1], minval=low2, maxval=high2, dtype=tf.float32)),
    # 'skip': tf.random_normal([n_hidden_b1, n_hidden_b1])
    'skip': tf.Variable(tf.random_normal([n_hidden_b1, n_hidden_b1]))
}

biases_block1 = {
    'layer1': tf.Variable(tf.random_normal([n_hidden_b1])),
    'layer2': tf.Variable(tf.random_normal([n_hidden_b1])),
    'layer3': tf.Variable(tf.random_normal([n_hidden_b1]))
}

low3 = -4*np.sqrt(6.0/(n_hidden_b1 + n_hidden_b2)) # use 4 for sigmoid, 1 for tanh activation
high3 = 4*np.sqrt(6.0/(n_hidden_b1 + n_hidden_b2))
low4 = -4*np.sqrt(6.0/(n_hidden_b2 + n_hidden_b2))
high4 = 4*np.sqrt(6.0/(n_hidden_b2 + n_hidden_b2))

weights_block2 = {
    'layer1': tf.Variable(tf.random_uniform([n_hidden_b1, n_hidden_b2], minval=low3, maxval=high3, dtype=tf.float32)),
    'layer2': tf.Variable(tf.random_uniform([n_hidden_b2, n_hidden_b2], minval=low4, maxval=high4, dtype=tf.float32)),
    'layer3': tf.Variable(tf.random_uniform([n_hidden_b2, n_hidden_b2], minval=low4, maxval=high4, dtype=tf.float32)),
    # 'skip': tf.random_normal([n_hidden_b2, n_hidden_b2])
    'skip': tf.Variable(tf.random_normal([n_hidden_b2, n_hidden_b2]))
}

biases_block2 = {
    'layer1': tf.Variable(tf.random_normal([n_hidden_b2])),
    'layer2': tf.Variable(tf.random_normal([n_hidden_b2])),
    'layer3': tf.Variable(tf.random_normal([n_hidden_b2]))
}

low5 = -4*np.sqrt(6.0/(n_hidden_b2 + n_hidden_b3)) # use 4 for sigmoid, 1 for tanh activation
high5 = 4*np.sqrt(6.0/(n_hidden_b2 + n_hidden_b3))
low6 = -4*np.sqrt(6.0/(n_hidden_b3 + n_hidden_b3))
high6 = 4*np.sqrt(6.0/(n_hidden_b3 + n_hidden_b3))

weights_block3 = {
    'layer1': tf.Variable(tf.random_uniform([n_hidden_b2, n_hidden_b3], minval=low5, maxval=high5, dtype=tf.float32)),
    'layer2': tf.Variable(tf.random_uniform([n_hidden_b3, n_hidden_b3], minval=low6, maxval=high6, dtype=tf.float32)),
    'layer3': tf.Variable(tf.random_uniform([n_hidden_b3, n_hidden_b3], minval=low6, maxval=high6, dtype=tf.float32)),
    # 'skip': tf.random_normal([n_hidden_b2, n_hidden_b2])
    'skip': tf.Variable(tf.random_normal([n_hidden_b3, n_hidden_b3]))
}

biases_block3 = {
    'layer1': tf.Variable(tf.random_normal([n_hidden_b3])),
    'layer2': tf.Variable(tf.random_normal([n_hidden_b3])),
    'layer3': tf.Variable(tf.random_normal([n_hidden_b3]))
}


# constructing the model
low = -4*np.sqrt(6.0/(n_hidden_b3 + n_categories)) # use 4 for sigmoid, 1 for tanh activation
high = 4*np.sqrt(6.0/(n_hidden_b3 + n_categories))
last_layer_weights = tf.Variable(tf.random_uniform([n_hidden_b3, n_categories], minval=low, maxval=high, dtype=tf.float32))
last_layer_biases = tf.Variable(tf.random_normal([n_categories]))

model = residual_block(x, weights_block1, biases_block1)
model = residual_block(model, weights_block2, biases_block2)
model = residual_block(model, weights_block3, biases_block3)
model = tf.add(tf.matmul(model, last_layer_weights), last_layer_biases)

# defining the cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# defining session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()

# initializing parameters
sess.run(init)

 # Training cycle
for epoch in range(number_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step

    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    train_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    train_accuracy = tf.reduce_mean(tf.cast(train_prediction, "float"))
    print("train_Accuracy:", train_accuracy.eval({x: mnist.train.images, y: mnist.train.labels}))

    valid_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, "float"))
    print("valid_Accuracy:", valid_accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels}))

print("Optimization Finished!")

# Test model
correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


