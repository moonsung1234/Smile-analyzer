
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import numpy as np
import datetime
import random
import math
import os

class ImageCnnModel :
    def __init__(self, learning_rate, loop_count, learning_number, inputs, outputs) :
        tf.disable_eager_execution()

        self.learning_rate = learning_rate
        self.loop_count = loop_count
        self.learning_number = learning_number
        self.input = inputs
        self.output = outputs

    def __setVariables(self) :
        self.x = tf.placeholder(tf.float32, [None, self.input], name="x")
        self.t = tf.placeholder(tf.float32, [None, self.output], name="t")
        
        self.reshape_wh = int(math.sqrt(self.input))

        self.a1 = tf.reshape(self.x, [-1, self.reshape_wh, self.reshape_wh, 1], name="a1")
        self.f1 = tf.Variable(tf.random_normal([3, 3, 1, 30], stddev=0.01), name="f1")
        self.b1 = tf.Variable(tf.constant(0.1, shape=[30]), name="b1")
        self.c1 = tf.nn.conv2d(self.a1, self.f1, strides=[1, 1, 1, 1], padding="SAME", name="c1")
        self.z1 = tf.nn.relu(self.c1 + self.b1, name="z1")

        self.a2 = tf.nn.max_pool(self.z1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="a2")
        self.f2 = tf.Variable(tf.random_normal([3, 3, 30, 60], stddev=0.01), name="f2")
        self.b2 = tf.Variable(tf.constant(0.1, shape=[60]), name="b2")
        self.c2 = tf.nn.conv2d(self.a2, self.f2, strides=[1, 1, 1, 1], padding="SAME", name="c2")
        self.z2 = tf.nn.relu(self.c2 + self.b2, name="z2")

        self.a3 = tf.nn.max_pool(self.z2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="a3")
        self.f3 = tf.Variable(tf.random_normal([3, 3, 60, 90], stddev=0.01), name="f3")
        self.b3 = tf.Variable(tf.constant(0.1, shape=[90]), name="b3")
        self.c3 = tf.nn.conv2d(self.a3, self.f3, strides=[1, 1, 1, 1], padding="SAME", name="c3")
        self.z3 = tf.nn.relu(self.c3 + self.b3, name="z3")

        self.a4 = tf.nn.max_pool(self.z3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="a4")

        self.a4_flat = tf.reshape(self.a4, [-1, 90 * int(self.reshape_wh / 8)**2], name="a4_flat")

        self.w = tf.Variable(tf.random_normal([90 * int(self.reshape_wh / 8)**2, self.output], stddev=0.01), name="w")
        self.b = tf.Variable(tf.random_normal([self.output]), name="b")

        self.z_out = tf.matmul(self.a4_flat, self.w) + self.b
        self.y = tf.nn.softmax(self.z_out)

    def __setOptimizer(self) :
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.z_out, labels=self.t))
        self.optimi = tf.train.AdamOptimizer(self.learning_rate)
        self.train = self.optimi.minimize(self.loss)

    def __setAccuracy(self) :
        self.predict_value = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.predict_value, dtype=tf.float32))

    def __batch(self, x, t, size) :
        random_x = []
        random_t =[]

        for _ in range(size) :
            random_index = random.randint(0, len(x) - 1)

            random_x.append(x[random_index])
            random_t.append(t[random_index])

        return random_x, random_t

    def train(self, x_data, t_data) :
        self.__setVariables()
        self.__setOptimizer()
        self.__setAccuracy()

        self.saver = tf.train.Saver()
        self.new_saver = tf.train.Saver()

        with tf.Session() as self.sess :
            self.sess.run(tf.global_variables_initializer())

            checkpoint = tf.train.latest_checkpoint("data")

            if checkpoint :
                self.new_saver.restore(self.sess, checkpoint)

            x, t = self.x, self.t
            train_x, train_t = self.__batch(x_data, t_data, int(0.8 * len(x_data)))
            test_x, test_t = self.__batch(x_data, t_data, int(0.2 * len(x_data)))

            for i in range(self.loop_count) :
                for j in range(int(len(train_x) / self.learning_number)) :
                    batch_x, batch_t = self.__batch(train_x, train_t, self.learning_number)
                    loss_value, _ = self.sess.run([self.loss, self.train], feed_dict={x : batch_x, t : batch_t})

                    print("step : ", i, " ", "loss : ", loss_value)

            if self.loop_count != 0 :
                accuracy_value = self.sess.run(self.accuracy, feed_dict={x : test_x, t : test_t})
                print("accuracy : ", accuracy_value)
                
                self.saver.save(self.sess, "data/imagecnn")
            
            self.is_learn = True

    def predict(self, x_data) :
        with tf.Session() as self.sess :
            self.sess.run(tf.global_variables_initializer())

            checkpoint = tf.train.latest_checkpoint("data")

            if checkpoint :
                self.saver.restore(self.sess, checkpoint)

            x = self.x
            predicted_value = self.sess.run(self.y, feed_dict={x : x_data}) 

        return predicted_value
