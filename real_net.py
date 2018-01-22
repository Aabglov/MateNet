# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import os
import numpy as np
import random
import tensorflow as tf
import pickle
import time

# PATHS -- absolute
dir_path = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = "saved"
checkpoint_path = os.path.join(dir_path,SAVE_DIR)#,"mtg_rec_char_steps.ckpt")
CHECKPOINT_NAME = "dict_steps.ckpt"
LOG_DIR = "/tmp/tensorboard/gen"#os.path.join(dir_path,SAVE_DIR,"log")
model_path = os.path.join(dir_path,SAVE_DIR,CHECKPOINT_NAME)

num_examples = 10000
NUM_EPOCHS = 10000
LOG_EPOCH = 100
batch_size = 1000
batches_per_epoch = int(num_examples / batch_size)

# Initialize members of the herd
layer_1_dim = 64#128
layer_2_dim = 32#64
output_dim = 12
input_dim = int(2 * output_dim)
num_layers = 1#3
LEARNING_RATE = 0.001
ADAM_BETA = 0.5


input_dim = 784
layer_1_dim = 128
layer_2_dim = 64
output_dim = 10

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        #tf.summary.scalar('max', tf.reduce_max(var))
        #tf.summary.scalar('min', tf.reduce_min(var))
        #tf.summary.histogram('histogram', var)

graph = tf.Graph()
with graph.as_default():

    # Get dynamic batch_size
    # Not sure this is actually needed
    #batch_size = tf.shape(x)[0]

    # We can't initialize these variables to 0 - the network will get stuck.
    def init_weight(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        #return tf.get_variable(shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        return tf.Variable(initial)

    def init_bias(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        # return  tf.get_variable(shape=shape, initializer=tf.constant_initializer(0.1))
        return tf.Variable(initial)


    class GenLayer:
        def __init__(self, in_dim, out_dim, name, act=tf.sigmoid, summarize=False):
            self.name = name
            self.act = act
            self.summarize = summarize
            self.weights = init_weight([in_dim,out_dim])
            self.biases = init_bias([out_dim])

        def forward(self,input_tensor,activate=True):
            with tf.name_scope(self.name):
                pre_activations = tf.matmul(input_tensor, self.weights) + self.biases
                if activate:
                    activations = self.act(pre_activations)
                    if self.summarize:
                        tf.summary.histogram('activations', activations)
                    return activations
                else:
                    return pre_activations

    # Placeholders
    with tf.name_scope("input"):
        x_input = tf.placeholder(tf.float32, shape=[None, input_dim])
        y_input = tf.placeholder(tf.float32, shape=[None, output_dim])

    # Model
    with tf.name_scope("model"):
        #layer_1 = GenLayer(input_dim,   layer_1_dim, name="layer_1", act=tf.sigmoid, summarize=True)
        #layer_2 = GenLayer(layer_1_dim, layer_2_dim, name="layer_2", act=tf.sigmoid, summarize=True)
        #layer_3 = GenLayer(layer_2_dim, output_dim,  name="layer_3", act=tf.sigmoid, summarize=True)
        layer_1 = GenLayer(input_dim, output_dim, name="layer_1", act=tf.sigmoid, summarize=True)

    with tf.name_scope("forward"):
        #layer_1_out = layer_1.forward(x_input)
        #layer_2_out = layer_2.forward(layer_1_out)
        #layer_3_out = layer_3.forward(layer_2_out,activate=False)
        layer_out = layer_1.forward(x_input,activate=False)

    # Loss
    with tf.name_scope("loss"):
        #loss = tf.losses.mean_squared_error(y_input,layer_3_out)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=layer_out))
        tf.summary.scalar('loss', loss)

    # Backward Propagation
    with tf.name_scope('train'):
         collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
         train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=ADAM_BETA).minimize(loss,var_list=collection)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)

    # Initializing the variables
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    # 'Saver' op to save and restore all the variables
    #saver = tf.train.Saver()


print("Beginning Session")

#Running first session
with tf.Session(graph=graph) as sess:
    # Initialize variables
    #sess.run(init)
    sess.run(tf.global_variables_initializer())

    # try:
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    #     #saver.restore(sess, ckpt.model_checkpoint_path)
    #     print("Model restored from file: %s" % model_path)
    # except Exception as e:
    #     print("Model restore failed {}".format(e))

    # Training cycle
    already_trained = 0
    for epoch in range(already_trained,NUM_EPOCHS):
        avg_cost = 0
        for batch_i in range(batches_per_epoch):
            batch = mnist.train.next_batch(batch_size)

            start = time.time()

            # Run optimization op (backprop) and cost op (to get loss value)
            input_dict = {x_input:batch[0],
                         y_input:batch[1]}

            summary,cost,_ = sess.run([merged,loss, train_step], feed_dict=input_dict)

            end = time.time()
            train_writer.add_summary(summary, epoch)
            avg_cost += cost
        print("Epoch:", '{}'.format(epoch), "cost=" , "{}".format(avg_cost/batches_per_epoch), "time:", "{}".format(end-start))


        # # Display logs per epoch step
        # if epoch % LOG_EPOCH == 0:
        #     #train_writer.add_run_metadata(run_metadata, "step_{}".format(epoch))
        #     print("saving {}".format(epoch)) # Spacer
        #     save_path = saver.save(sess, model_path, global_step = epoch)

    # Save model weights to disk
    #save_path = saver.save(sess, model_path)
    #print("Model saved in file: %s" % save_path)
