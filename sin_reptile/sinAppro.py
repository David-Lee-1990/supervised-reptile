import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def gen_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    # print("This iteration's meta value(%f,%f)"%(phase,ampl))
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

seed = 0
rng = np.random.RandomState(seed)
tf.set_random_seed(seed)
x_all = np.linspace(-5,5,50)[:,None]
ntrain = 10 # size of training minibatches

_x = tf.placeholder(dtype=tf.float32, shape=(None,1))
_y = tf.placeholder(dtype=tf.float32, shape=(None,1))

N_inter = [64,64]

def weight(shape,flag="uniform"):

    if flag=="uniform":
        return (tf.Variable(tf.random_uniform(shape=shape,minval=-1/np.sqrt(shape[0]), maxval=1/np.sqrt(shape[0]),dtype=tf.float32),
                           name="w"),

                tf.Variable(tf.random_uniform(shape=[shape[1]], minval=-1 / np.sqrt(shape[0]), maxval=1 / np.sqrt(shape[0]),
                                              dtype=tf.float32),
                            name="b")
                )
    else:
        return (tf.Variable(tf.random_normal(shape=shape,dtype=tf.float32),
                           name="w"),
                tf.Variable(tf.random_normal(shape=[shape[1]], dtype=tf.float32),
                            name="b")
                )
flag="uniform"


with tf.variable_scope("MLP1"):

    w_1,b_1 = weight([1,N_inter[0]],flag)

    model = tf.nn.tanh(tf.matmul(_x,w_1)+b_1)

with tf.variable_scope("MLP2"):

    w_2,b_2 = weight([N_inter[0],N_inter[1]],flag)

    model = tf.nn.tanh(tf.matmul(model,w_2)+b_2)

with tf.variable_scope("MLP3"):

    w_3,b_3 = weight([N_inter[1],1],flag)

    model = tf.matmul(model, w_3)+b_3

loss = tf.reduce_mean(tf.pow(model - _y, 2))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
# AllVariables = tf.global_variables()
# gradients = tf.gradients(loss,AllVariables)
# values = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in AllVariables]
# assigns_0 = [tf.assign(AllVariables[n],values[n]) for n in range(len(AllVariables))]
# assigns = tf.group(*assigns_0)

def predict(x):
    return sess.run(model,feed_dict={_x:x})

f = gen_task()
xtrain = x_all[rng.choice(len(x_all), size=10)]
ytrain = f(xtrain)
plt.plot(x_all, f(x_all), label="true", color=(0, 1, 0))
plt.plot(xtrain, f(xtrain), "x", label="train", color="k")
plt.ylim(-4, 4)

niterations = 100
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for iteration in range(niterations):

        sess.run(train_step,feed_dict={_x: xtrain, _y: ytrain})
        # PreValues = sess.run(AllVariables)
        # gradients_ = sess.run(gradients,feed_dict={_x: xtrain,_y: ytrain})
        # AfterValues = np.array(PreValues) - 0.02*np.array(gradients_)
        # sess.run(assigns,feed_dict=dict(zip(values, AfterValues)))


        if iteration == 0 or (iteration + 1) % 50 == 0:
            # plt.cla()
            plt.plot(x_all, predict(x_all), label="pred after %d iters" % iteration,
                     color=(iteration / niterations, 0, 1 - iteration / niterations))
            # plt.pause(0.01)
    plt.legend(loc="upper left")
    plt.show()


