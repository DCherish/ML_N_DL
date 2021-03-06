#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import random


# In[4]:


tf.set_random_seed(777)


# In[5]:


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[6]:


learning_rate = 0.001
training_epochs = 15
batch_size = 100

keep_prob = tf.placeholder(tf.float32)


# In[7]:


X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])


# In[8]:


L1 = tf.layers.dense(inputs = X, units=512, activation = tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

L2 = tf.layers.dense(inputs = L1, units=512, activation = tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.dropout(L1, keep_prob=keep_prob)

L3 = tf.layers.dense(inputs = L2, units=512, activation = tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.dropout(L1, keep_prob=keep_prob)

L4 = tf.layers.dense(inputs = L3, units=512, activation = tf.nn.relu,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.dropout(L1, keep_prob=keep_prob)

hypothesis = tf.layers.dense(inputs = L4, units=10, activation = None,
                   kernel_initializer=tf.contrib.layers.xavier_initializer())


# In[9]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# In[10]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[11]:


for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print("Learning finished!")

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(
    "Accuracy: ", sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
print(
    "Prediction: ",
    sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1], keep_prob: 1})
)

plt.imshow(
    mnist.test.images[r : r + 1].reshape(28, 28),
    cmap="Greys",
    interpolation="nearest"
)
plt.show()


# In[ ]:




