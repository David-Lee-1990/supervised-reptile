import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from termcolor import colored

seed = 0
innerstepsize = 0.02
innerepochs = 1
outerstepsize0 = 0.1
niterations = 30000

rng = np.random.RandomState(seed)
tf.set_random_seed(seed)
x_all = np.linspace(-5,5,50)[:,None]
ntrain = 10 # size of training minibatches

def gen_task():
    phase = rng.uniform(low=0, high=2*np.pi)
    ampl = rng.uniform(0.1, 5)
    # print("This iteration's meta value(%f,%f)"%(phase,ampl))
    f_randomsine = lambda x: np.sin(x + phase) * ampl
    return f_randomsine

sess = tf.InteractiveSession()


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

# loss = tf.reduce_mean(tf.pow(model - _y, 2))+0.1*(regularizer_w1+regularizer_w2+regularizer_w3)
loss = tf.reduce_mean(tf.pow(model - _y, 2))
train_step = tf.train.GradientDescentOptimizer(innerstepsize).minimize(loss)
AllVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
get_gradients = tf.gradients(loss, AllVariables)
values = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in AllVariables]
assigns_0 = [tf.assign(AllVariables[n],values[n]) for n in range(len(AllVariables))]
assigns = tf.group(*assigns_0)

def train_on_batch(x,y):

    begin_0 = time.time()
    Variables_values = sess.run(AllVariables)
    gradients = sess.run(get_gradients, feed_dict={_x:x,_y:y})
    end_0 = time.time()
    # print("Gradient Calculation cost %f seconds"%(end_0-begin_0))

    New_values = Variables_values - innerstepsize * np.array(gradients)
    begin = time.time()
    # sess.run(assigns,feed_dict={values:New_values})
    sess.run(assigns, feed_dict=dict(zip(values, New_values)))
    end = time.time()
    # print(colored("Re evaluation cost %f seconds"%(end-begin),"green"))
    # print(colored("train on batch total costs %f seconds"%(end-begin_0),"red"))
    # sess.run(train_step,feed_dict={_x:x,_y:y})

def predict(x):
    return sess.run(model, feed_dict={_x: x})

sess.run(tf.global_variables_initializer())
tf.get_default_graph().finalize()

f_plot = gen_task()
xtrain_plot = x_all[rng.choice(len(x_all), size=ntrain)]

old_variables = sess.run(AllVariables)


for iteration in range(niterations):
    print("Iteration %d"%iteration)
    begin_0 = time.time()
    f = gen_task()
    y_all = f(x_all)
    indexs = rng.permutation(len(x_all))
    # new_variables = []
    begin_1 = time.time()
    for _ in range(innerepochs):
        for start in range(0, len(x_all), ntrain):
            mbinds = indexs[start:start+ntrain]
            train_on_batch(x_all[mbinds], y_all[mbinds])
            # new_variables.append(sess.run(tf.global_variables()))
            # print(sess.run(loss,feed_dict={_x:x_all[mbinds],_y:y_all[mbinds]}))
    end_1 = time.time()
    tem_variables = sess.run(AllVariables)
    # print((np.array(old_variables)==np.array(tem_variables)))
    # tem_variables = np.sum(new_variables,0)/len(new_variables)
    old_variables += outerstepsize0*(1 - iteration/niterations)*(np.array(tem_variables)-np.array(old_variables))

    sess.run(assigns, feed_dict=dict(zip(values,old_variables)))
    # for n in range(len(old_variables)):
    #     tf.assign(AllVariables[n], old_variables[n])

    end_0 = time.time()
    print("This Iteration's training part costs %f seconds" % (end_1-begin_1))
    print("This Iteration costs %f seconds" % (end_0 - begin_0))
    if iteration == 0 or (iteration+1) % 1000 == 0:
        plt.cla()
        f = f_plot
        weights_before = deepcopy(old_variables)
        # print(predict(x_all))
        plt.plot(x_all, predict(x_all), label="pred after 0", color=(0, 0, 1))
        for inneriter in range(32):
            train_on_batch(xtrain_plot, f(xtrain_plot))
            if (inneriter+1)%16==0:
                frac = (inneriter+1)/32
                plt.plot(x_all,predict(x_all),label="pred after %d"%(inneriter+1),color=(frac,0,1-frac))
        plt.plot(x_all,f(x_all),label="true",color=(0,1,0))
        plt.plot(xtrain_plot, f(xtrain_plot), "x", label="train", color="k")
        plt.legend(loc='upper right')
        plt.show()
        sess.run(assigns, feed_dict=dict(zip(values, weights_before)))
sess.close()






