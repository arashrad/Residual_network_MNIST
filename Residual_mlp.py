# A 30-layers MLP with skip connections between non consecutive layers.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


learning_rate = 0.001
number_epochs = 20
batch_size = 100


def main():
    """This is the main training function"""

    # we start by loading MNIST data-set
    # reading the saved data-set from HDD
    f = open("augmented_dataset", "rb")
    x_train = np.load(f)
    y_train = np.load(f)
    x_valid = np.load(f)
    y_valid = np.load(f)
    x_test = np.load(f)
    y_test = np.load(f)
    f.close()
    print("data is loaded...")

    sess = tf.InteractiveSession()

    # defining input and output placeholders
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')

    # initializing weight variables
    def weights_var(shape, act=tf.nn.sigmoid):
        """initializing weights for sigmoid activation function based on Xavier initialization"""
        fan_in = shape[0]
        fan_out = shape[1]
        if act == tf.nn.sigmoid:  # in the case of sigmoid as activation function
            low = -4 * np.sqrt(6.0 / (fan_in + fan_out))
            high = 4 * np.sqrt(6.0 / (fan_in + fan_out))
            return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
        else:  # in the case of Relu or any other activation functions
            low = -np.sqrt(2.0 / (fan_in + fan_out))
            high = np.sqrt(2.0 / (fan_in + fan_out))
            return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

    def biases_var(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def neural_network_layer(layer_input, num_in, num_out, act=tf.nn.sigmoid):
        """this is building block of each hidden layer in MLP network"""
        layer_weights = weights_var([num_in, num_out])
        layer_biases = biases_var([num_out])
        pre_activation = tf.matmul(layer_input, layer_weights) + layer_biases
        activations = act(pre_activation, name='activation')
        return activations

    def residual_blocks(layer_input, num_in,  block_shape):
        if num_in == block_shape[1]:
            layer_1 = neural_network_layer(layer_input, num_in=num_in, num_out=block_shape[0])
            layer_2 = neural_network_layer(layer_1, num_in=block_shape[0], num_out=block_shape[1], act=tf.identity)
            # skip_value = tf.matmul(layer_input, weights_var([num_in, block_shape[1]]))
            output = tf.nn.sigmoid(tf.add(layer_2, layer_input))
            return output
        else:
            layer_1 = neural_network_layer(layer_input, num_in=num_in, num_out=block_shape[0])
            layer_2 = neural_network_layer(layer_1, num_in=block_shape[0], num_out=block_shape[1], act=tf.identity)
            skip_value = tf.matmul(layer_input, weights_var([num_in, block_shape[1]]))
            # output = tf.nn.sigmoid(tf.add(layer_1, layer_3))
            output = tf.nn.sigmoid(tf.add(layer_2, skip_value))
            return output

    # Now we define last layer(output layer) but without Softmax,
    # so we pass an identity argument to the neural_network_layer constructor
    # as the activation function
    first_layer = neural_network_layer(x, 784, 256)
    model = residual_blocks(first_layer, 256, [128, 128])
    model = residual_blocks(model, 128, [64, 64])
    for j in range(12):
        model = residual_blocks(model, 64, [64, 64])
    y_hat = neural_network_layer(model, 64, 10, act=tf.identity)

    # we define cross entropy loss function
    with tf.name_scope('cross_entropy'):
        delta = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(delta)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # training
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    tf.global_variables_initializer().run()

    loss_records = []
    validation_records = []
    training_records = []
    test_record = []
    for t in range(number_epochs):
        # shuffling data for each epoch
        x_train, y_train = shuffle(x_train, y_train)

        print("Start training at epoch: ", t+1)
        for i in range(x_train.shape[0] // batch_size):
            # Train the network and record training summery
            x_train_batch = x_train[i*batch_size:i*batch_size + batch_size]
            y_train_batch = y_train[i*batch_size:i * batch_size + batch_size]
            _ = sess.run(train_step, feed_dict={x: x_train_batch, y: y_train_batch})
            # train_writer.add_summary(summary, i)
            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict={x: x_valid, y: y_valid})
                print('Accuracy at step %s: %s' % (i, acc))
                validation_records.append(acc)
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: x_train, y: y_train})
                training_records.append(acc)
                loss_records.append(loss)

    acc = sess.run([accuracy], feed_dict={x: x_test, y: y_test})
    print('Training is finshed and test set Accuracy is: %s' %(acc))
    test_record.append(acc)

    # saving records
    f = open("saved_records_resnet_identity_30_v2", "wb")
    np.save(f, np.array(loss_records))
    np.save(f, np.array(training_records))
    np.save(f, np.array(validation_records))
    np.save(f, np.array(test_record))
    f.close()
    print("data is saved...")

main()







