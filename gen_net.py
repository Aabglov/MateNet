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
batch_size = 10000
batches_per_epoch = int(num_examples / batch_size)

layer_1_dim = 128
layer_2_dim = 64
output_dim = 12
input_dim = int(2 * output_dim)
num_layers = 3

weight_shape_1 = (input_dim,layer_1_dim)
bias_shape_1 = (layer_1_dim)
weight_shape_2 = (layer_1_dim,layer_2_dim)
bias_shape_2 = (layer_2_dim)
weight_shape_3 = (layer_2_dim,output_dim)
bias_shape_3 = (output_dim)

def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x,dtype="float32")
    y = np.array(y,dtype="float32")

    return (x,y)

def generateWeights(shape):
    return np.random.normal(size=shape,scale=0.1)

# For now this function is the same as generateWeights,
# but I may want to change that in the future
def generateBiases(shape):
    return np.random.normal(size=shape,scale=0.1)

class Individual:
    # Expecting list of weight matrices
    # and list of biases vectors
    def __init__(self,weights,biases):
        self.weights = weights
        self.biases = biases
        self.loss = 0
        self.generation = 0

def createNewIndividual():
    weights = [generateWeights(weight_shape_1),
               generateWeights(weight_shape_2),
               generateWeights(weight_shape_3)]

    biases = [generateBiases(bias_shape_1),
              generateBiases(bias_shape_2),
              generateBiases(bias_shape_3)]

    return Individual(weights,biases)

# Initialize members of the herd
num_children = 10
colony = []
for c in range(num_children):
    colony.append(createNewIndividual())


def mutate(mat,mutation_factor=0.003):
    return mat + np.random.normal(size=mat.shape,scale=mutation_factor)

# For now mating only involves 2 individuals,
# but if the results are promising I may include
# a "kink" variable that allows more than 1 partner.
#....
# Oh my
def Mate(indiv1,indiv2):
    weights = []
    biases = []
    for i in range(num_layers):
        avg_weights = (indiv1.weights[i] + indiv2.weights[i]) / 2.0
        avg_biases = (indiv1.biases[i] + indiv2.biases[i]) / 2.0
        weights.append(avg_weights)
        biases.append(avg_biases)
    new_colony = []
    for i in range(num_children):
        mutated_weights = [mutate(w) for w in weights]
        mutated_biases = [mutate(b) for b in biases]
        new_indiv = Individual(mutated_weights,mutated_biases)
        new_indiv.generation += 1
        new_colony.append(new_indiv)
    return new_colony


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


    def GenLayer(input_tensor, weights, biases, layer_name, act=tf.sigmoid, summarize=False):
        with tf.name_scope(layer_name):
            pre_activations = tf.matmul(input_tensor, weights) + biases
            activations = act(pre_activations)
            if summarize:
                tf.summary.histogram('activations', activations)
        return activations

    # Placeholders
    with tf.name_scope("input"):
        x_input = tf.placeholder(shape=[None, input_dim], dtype=tf.float32, name='x_input')
        y_input = tf.placeholder(shape=[None, output_dim],dtype=tf.float32, name='y_input')
        # For now we're assuming a 3-layer network.
        # In the future I'll change this to be dynamic as well.
        #   Layer 1
        weights_1_input = tf.placeholder(shape=[input_dim,layer_1_dim], dtype=tf.float32, name='weights_1')
        biases_1_input = tf.placeholder(shape=[layer_1_dim], dtype=tf.float32, name='biases_1')
        # Layer 2
        weights_2_input = tf.placeholder(shape=[layer_1_dim,layer_2_dim], dtype=tf.float32, name='weights_2')
        biases_2_input = tf.placeholder(shape=[layer_2_dim], dtype=tf.float32, name='biases_2')
        # Layer 3
        weights_3_input = tf.placeholder(shape=[layer_2_dim,output_dim], dtype=tf.float32, name='weights_3')
        biases_3_input = tf.placeholder(shape=[output_dim], dtype=tf.float32, name='biases_3')

    # Model
    with tf.name_scope("model"):
        layer_1_out = GenLayer(x_input,     weights_1_input, biases_1_input, "layer_1", act=tf.sigmoid, summarize=True)
        layer_2_out = GenLayer(layer_1_out, weights_2_input, biases_2_input, "layer_2", act=tf.sigmoid, summarize=True)
        layer_3_out = GenLayer(layer_2_out, weights_3_input, biases_3_input, "layer_3", act=tf.sigmoid, summarize=True)

    # Loss
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(y_input,layer_3_out)
        tf.summary.scalar('loss', loss)

    # Backward Propagation
    # with tf.name_scope('train'):
    #     collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #     train_step = tf.train.AdamOptimizer(LEARNING_RATE,beta1=ADAM_BETA).minimize(loss,var_list=collection)

    # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'train'), graph)

    # Initializing the variables
    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()

    # 'Saver' op to save and restore all the variables
    #saver = tf.train.Saver()


print("Beginning Session")
x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)


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
        for batch_i in range(batches_per_epoch):
            batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
            batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]

            start = time.time()
            avg_cost = 0
            for individual in colony:
                # Run optimization op (backprop) and cost op (to get loss value)
                input_dict = {x_input:batch_x,
                             y_input:batch_y,
                             weights_1_input:individual.weights[0],
                             biases_1_input:individual.biases[0],
                             weights_2_input:individual.weights[1],
                             biases_2_input:individual.biases[1],
                             weights_3_input:individual.weights[2],
                             biases_3_input:individual.biases[2]
                }

                summary,cost = sess.run([merged,loss], feed_dict=input_dict)
                individual.loss = cost
                avg_cost += cost

            end = time.time()
            train_writer.add_summary(summary, epoch)
            print("Epoch:", '{}'.format(epoch), "cost=" , "{}".format(avg_cost/num_children), "time:", "{}".format(end-start))

            # Determine the 2 best-performing individuals
            # and mate them to create next generation
            colony_sorted_by_cost = sorted(colony, key=lambda x: x.loss, reverse=False)
            best_1 = colony_sorted_by_cost[0]
            best_2 = colony_sorted_by_cost[1]
            colony = Mate(best_1,best_2)

        # # Display logs per epoch step
        # if epoch % LOG_EPOCH == 0:
        #     #train_writer.add_run_metadata(run_metadata, "step_{}".format(epoch))
        #     print("saving {}".format(epoch)) # Spacer
        #     save_path = saver.save(sess, model_path, global_step = epoch)

    # Save model weights to disk
    #save_path = saver.save(sess, model_path)
    #print("Model saved in file: %s" % save_path)
