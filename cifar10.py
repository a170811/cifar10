import tensorflow as tf
from tensorflow.contrib.layers import flatten
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = shuffle(x_train, y_train)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,
        test_size = 0.1)

# Declare variables
batch_size = 48
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
epochs = 30 # repeat 100 times
num_classes = 10
rate = 0.001
#label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
#        6: "frog", 7: "horse", 8: "ship", 9: "truck"}
#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
#        'ship', 'truck']


x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_validation  /= 255
x_test /= 255



def LeNet(x):

    mu = 0
    sigma = 0.1

    def add_conv(prev, shape, padding, bn, act, max_pool, drop, keep_rate = None):
        with tf.name_scope('Layer'):
            with tf.name_scope('Weight'):
                conv_W = tf.Variable(tf.truncated_normal(shape = shape, mean = mu, stddev = sigma))
            with tf.name_scope('Biases'):
                conv_b = tf.Variable(tf.zeros(shape[3]))
            conv = tf.nn.conv2d(prev, conv_W, strides = [1, 1, 1, 1], padding = padding) + conv_b
            if bn:
                conv = tf.layers.batch_normalization(conv, training = is_training)
            if act:
                conv = tf.nn.relu(conv)
            if drop:
                conv = tf.nn.dropout(conv, keep_rate)
            if max_pool:
                conv = tf.nn.max_pool(conv, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
        return conv


    #input: 32*32*3 , output: 28*28*6
    #add_conv(prev, shape, padding, bn, act, max_pool, keep_rate):
    conv1 = add_conv(x, (3, 3, 3, 6), 'SAME', True, True, False, True, keep_prob)
    conv2 = add_conv(conv1, (3, 3, 6, 16), 'SAME', True, True, True, False)
    conv3 = add_conv(conv2, (3, 3, 16, 32), 'VALID', True, True, True, False)
    conv4 = add_conv(conv3, (3, 3, 32, 64), 'SAME', True, True, False, True, keep_prob)
    conv5 = add_conv(conv4, (4, 4, 64, 64), 'VALID', True, True, True, False)


    with tf.name_scope('Layer'):
        fc0 = flatten(conv5)

        with tf.name_scope('Weight'):
            fc1_W = tf.Variable(tf.truncated_normal(shape = (256, 256), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            fc1_b = tf.Variable(tf.zeros(256))
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.layers.batch_normalization(fc1, training = is_training)
        fc1 = tf.nn.dropout(fc1, keep_prob)
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('Layer'):
        with tf.name_scope('Weight'):
            fc2_W = tf.Variable(tf.truncated_normal(shape = (256, 10), mean = mu, stddev = sigma))
        with tf.name_scope('Biases'):
            fc2_b = tf.Variable(tf.zeros(10))
        logits = tf.matmul(fc1, fc2_W) + fc2_b
        logits = tf.layers.batch_normalization(logits, training = is_training)
        logits = tf.nn.dropout(logits, keep_prob)

    return logits


###training model
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
one_hot_y = tf.squeeze(tf.one_hot(y, 10))
x = tf.image.random_flip_left_right(x)
x = tf.image.random_brightness(x, max_delta = 0.5)
x = tf.image.random_contrast(x, 0.1, 0.6)


logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.stop_gradient(one_hot_y), logits = logits)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss)



###accuracy
with tf.name_scope('acc'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc', accuracy_operation)
saver = tf.train.Saver()


###training
with tf.Session() as sess:
    steps = 0
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("train/", graph = sess.graph)
    test_writer = tf.summary.FileWriter("test/", graph = sess.graph)

    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)

    file = open('./out.txt', 'w')
    file.write('Training...\n')
    file.write('\n')
    for i in range(epochs):
        print('epoch', i + 1)
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]

            if steps%200 == 0:
                train_result = sess.run(merged, feed_dict = {x: batch_x, y: batch_y,
                        keep_prob: 1, is_training: False})
                train_writer.add_summary(train_result, steps)


                test_result = sess.run(merged, feed_dict = {x: x_validation, y: y_validation,
                        keep_prob: 1, is_training: False})
                test_writer.add_summary(test_result, steps)

            steps += 1
            sess.run(training_operation, feed_dict = {x: batch_x, y: batch_y,
                        keep_prob:0.8, is_training: True})

        testing_accuracy = sess.run(accuracy_operation, feed_dict = {x: x_test, y: y_test,
                keep_prob: 1, is_training:False})

        file.write('EPOCH {} ...\n'.format(i + 1))
        file.write('Testing Accuracy = {:.3f}\n'.format(testing_accuracy))
        file.write('\n')

    saver.save(sess, './cifar10')
    file.write('Model saved\n')
    file.close()
