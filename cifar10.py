import tensorflow as tf
from tensorflow.contrib.layers import flatten
import random
import numpy as np
from sklearn.utils import shuffle

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Declare variables
batch_size = 48
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
epochs = 20 # repeat 100 times
num_classes = 10
rate = 0.001
#label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse",
#                  8: "ship", 9: "truck"}
#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255


def LeNet(x):

    mu = 0
    sigma = 0.1

    #input: 32*32*3 , output: 28*28*6
    with tf.name_scope('Layer'):
        with tf.name_scope('Weight'):
            conv1_W = tf.Variable(tf.truncated_normal(shape = (5, 5, 3, 6), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x, conv1_W, strides = [1, 1, 1, 1], padding = 'VALID') + conv1_b

        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')


    with tf.name_scope('Layer'):
        with tf.name_scope('Weight'):
            conv2_W = tf.Variable(tf.truncated_normal(shape = (5, 5, 6, 16), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides = [1, 1, 1, 1], padding = 'VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')


    with tf.name_scope('Layer'):
        fc0 = flatten(conv2)

        with tf.name_scope('Weight'):
            fc1_W = tf.Variable(tf.truncated_normal(shape = (400, 120), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('Layer'):
        with tf.name_scope('Weight'):
            fc2_W = tf.Variable(tf.truncated_normal(shape = (120, 84), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)

    with tf.name_scope('Layer'):
        with tf.name_scope('Weight'):
            fc3_W = tf.Variable(tf.truncated_normal(shape = (84, 10), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            fc3_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


###training model
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.squeeze(tf.one_hot(y, 10))



logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.stop_gradient(one_hot_y), logits = logits)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss)



###accuracy
with tf.name_scope('acc'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc', accuracy_operation)
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


###training
with tf.Session() as sess:
    steps = 0
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)

    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    file = open('./out.txt', 'w')
    file.write('Training...\n')
    file.write('\n')
    for i in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)
        #x_train = tf.image.random_flip_left_right(x_train)
        #x_train = tf.image.random_brightness(x_train, max_delta = 0.5)
        #x_train = tf.image.random_contrast(x_train, 0.1, 0.6).eval()
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]

            if steps%200 == 0:
                result = sess.run(merged, feed_dict = {x: batch_x, y: batch_y})
                writer.add_summary(result, steps)
            steps += 1
            sess.run(training_operation, feed_dict = {x: batch_x, y: batch_y})
        validation_accuracy = sess.run(accuracy_operation, feed_dict = {x: x_test, y: y_test})
        file.write('EPOCH {} ...\n'.format(i + 1))
        file.write('Validation Accuracy = {:.3f}\n'.format(validation_accuracy))
        file.write('\n')

    saver.save(sess, './cifar10')
    file.write('Model saved\n')
    file.close()
