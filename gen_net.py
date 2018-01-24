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
LEARNING_RATE = 0.01

# Initialize members of the herd
num_children = 100#20
num_sub_optimal = int(num_children / 10.0)
layer_1_dim = 128
layer_2_dim = 64
output_dim = 10
input_dim = 784
num_layers = 1

weight_shape_1 = (input_dim,output_dim)
bias_shape_1 = (output_dim)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def generateWeights(shape):
    return np.random.normal(size=shape,scale=0.1)
    #return np.random.randn(*shape)

# For now this function is the same as generateWeights,
# but I may want to change that in the future
def generateBiases(shape):
    return np.random.normal(size=shape,scale=0.1)
    #return np.random.randn(shape)

class Individual:
    # Expecting list of weight matrices
    # and list of biases vectors
    def __init__(self,weights,biases):
        self.weights = weights
        self.biases = biases
        self.loss = 0
        self.generation = 0

def createNewIndividual():
    weights = [generateWeights(weight_shape_1)]
    biases = [generateBiases(bias_shape_1)]

    return Individual(weights,biases)


colony = []
for c in range(num_children):
    colony.append(createNewIndividual())


def mutate(mat,mutation_factor=0.01):
    return mat + np.random.normal(size=mat.shape,scale=mutation_factor)



# For now mating only involves 2 individuals,
# but if the results are promising I may include
# a "kink" variable that allows more than 1 partner.
#....
# Oh my
def Mate(indiv1,indiv2,mutation_factor=0.01):
    weights = []
    biases = []
    for i in range(len(indiv1.weights)):
        avg_weights = (indiv1.weights[i] + indiv2.weights[i]) / 2.0
        avg_biases = (indiv1.biases[i] + indiv2.biases[i]) / 2.0
        weights.append(avg_weights)
        biases.append(avg_biases)
    new_colony = []
    for i in range(num_children-num_sub_optimal):
        mutated_weights = [mutate(w,mutation_factor) for w in weights]
        mutated_biases = [mutate(b,mutation_factor) for b in biases]
        new_indiv = Individual(mutated_weights,mutated_biases)
        new_indiv.generation += 1
        new_colony.append(new_indiv)
    for _ in range(num_sub_optimal):
        new_colony.append(createNewIndividual())
    return new_colony

# This method simulates asexual reproduction
# with mutation
def MonoReproduce(indiv,mutation_factor=0.01):
    mutated_weights = [mutate(w,mutation_factor) for w in indiv.weights]
    mutated_biases = [mutate(b,mutation_factor) for b in indiv.biases]
    new_indiv = Individual(mutated_weights,mutated_biases)
    new_indiv.generation += 1

    return new_indiv

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

    # Mutate/"Learn"
    def mutateModel(model,mutation_factor=0.1):
        model.weights += np.random.normal(size=model.weights.shape,scale=mutation_factor)
        model.biases  += np.random.normal(size=model.biases.shape, scale=mutation_factor)
        return model

    class GenLayer:
        def __init__(self, in_dim, out_dim, name, act=tf.sigmoid, summarize=False):
            self.name = name
            self.act = act
            self.in_dim = in_dim
            self.out_dim = out_dim
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
        to_repro_list = tf.placeholder(tf.int32, shape=[num_children])

    # Model
    with tf.name_scope("model"):
        models = []
        for i in range(num_children):
            models.append(GenLayer(input_dim, output_dim, name="model_{}".format(i), act=tf.sigmoid, summarize=True))

    with tf.name_scope("forward"):
        outputs = []
        for i in range(num_children):
            outputs.append(models[i].forward(x_input,activate=False))

    # Loss
    with tf.name_scope("loss"):
        losses = []
        #loss = 0.
        for i in range(num_children):
            l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=outputs[i]))
            losses.append(l)
            #loss += l
        #loss /= num_children
        #tf.summary.scalar('loss', loss)

    # Training
    with tf.name_scope("reproduce"):
        new_models = []
        with tf.Session() as sess:
            for i in range(num_children):
                index = tf.cast(to_repro_list[i],np.int32)
                new_models.append(mutateModel(models[index.eval()]))
        models = new_models
        # This value isn't used, it's
        # just here to force the iteration
        # to go through this step
        repro_trigger = tf.reduce_sum(models[0].weights)

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

        start = time.time()
        for batch_i in range(batches_per_epoch):
            batch = mnist.train.next_batch(batch_size)

            avg_batch_cost = 0
            cost_list = []
            repro_list = np.array([i for i in range(num_children)])

            input_dict = {x_input:batch[0],
                         y_input:batch[1],
                         to_repro_list:repro_list
            }

            costs,_ = sess.run([losses,repro_trigger], feed_dict=input_dict)

            # for individual in colony:
            #     # Run optimization op (backprop) and cost op (to get loss value)
            #     input_dict = {x_input:batch[0],
            #                  y_input:batch[1]
            #     }
            #
            #     summary,cost = sess.run([merged,loss], feed_dict=input_dict)
            #     individual.loss = cost
            #     cost_list.append(cost)
            #     avg_cost += cost
            #     avg_batch_cost += cost
            #
            # avg_batch_cost /= num_children
            #
            # inverse_cost_list = [1./c for c in cost_list]
            # inverse_cost_total = sum(inverse_cost_list)
            # inverse_cost_list /= inverse_cost_total
            # num_costs = len(inverse_cost_list)
            # #print(inverse_cost_list)
            # indivs_to_repro = np.random.choice(num_costs, num_costs, p=inverse_cost_list)
            # #print(vals)
            # new_colony = []
            # for i in indivs_to_repro:
            #     indiv = colony[i]
            #     new_colony.append(MonoReproduce(indiv,LEARNING_RATE))
            # colony = new_colony

            # # Determine the 2 best-performing individuals
            # # and mate them to create next generation
            # colony_sorted_by_cost = sorted(colony, key=lambda x: x.loss, reverse=False)
            # best_1 = colony_sorted_by_cost[0]
            # best_2 = colony_sorted_by_cost[1]
            # colony = Mate(best_1,best_2,LEARNING_RATE)#/30.)
        avg_cost = np.mean(costs)
        end = time.time()
        print("Epoch:", '{}'.format(epoch), "cost=" , "{}".format(avg_cost), "time:", "{}".format(end-start))

        # # Display logs per epoch step
        # if epoch % LOG_EPOCH == 0:
        #     #train_writer.add_run_metadata(run_metadata, "step_{}".format(epoch))
        #     print("saving {}".format(epoch)) # Spacer
        #     save_path = saver.save(sess, model_path, global_step = epoch)

    # Save model weights to disk
    #save_path = saver.save(sess, model_path)
    #print("Model saved in file: %s" % save_path)
