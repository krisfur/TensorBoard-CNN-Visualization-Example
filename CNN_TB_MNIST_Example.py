#----------------------------------------------------------------------
# MNIST data classifier using a Conolutional Neural Network.
# 
# Author: Krzysztof Furman; 15.08.2017
# TensorBoard support:
#   :scalars:
#     - accuracy
#     - wieghts and biases
#     - cost/cross entropy
#     - dropout
#   :images:
#     - reshaped input
#     - conv layers outputs
#     - conv layers weights visualisation
#   :graph:
#     - full graph of the network
#   :distributions and histograms:
#     - weights and biases
#     - activations
#   :checkpoint saving:
#     - checkpoints/saving model
#     - weights embeddings
#
#   :to be implemented: 
#     - image embeddings (as in https://www.tensorflow.org/get_started/embedding_viz)
#     - ROC curve calculation (as in http://blog.csdn.net/mao_feng/article/details/54731098)
#----------------------------------------------------------------------
import tensorflow as tf
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###READ DATA###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
print ("\nImporting the MNIST data")
mnist = read_data_sets("/tmp/data/", one_hot=True)
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###SET THE CNN OPTIONS###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
n_outputs= 10
image_x  = 28
image_y  = 28
display_step = 10
training_epochs = 200
image_shape = [-1, image_x, image_y, 1]
batch_size = 50
learning_rate = 1e-4
#n_features = image_x*image_y
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###BUILD THE CNN###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# set placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
#----------------------------------------------------------------------
with tf.name_scope('input_reshape'):
    x_reshaped = tf.reshape(x, image_shape)
    tf.summary.image('input', x_reshaped, 10)
#----------------------------------------------------------------------
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
#----------------------------------------------------------------------
# First conv+pool layer
#----------------------------------------------------------------------
with tf.name_scope('conv1'):
    with tf.name_scope('weights'):
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(W_conv1)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv1 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(W_conv1))
            tf.summary.scalar('min', tf.reduce_min(W_conv1))
            tf.summary.histogram('histogram', W_conv1)
    
    with tf.name_scope('biases'):
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(b_conv1)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv1 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(b_conv1))
            tf.summary.scalar('min', tf.reduce_min(b_conv1))
            tf.summary.histogram('histogram', b_conv1)
    with tf.name_scope('Wx_plus_b'):
        preactivated1 = tf.nn.conv2d(x_reshaped, W_conv1,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv1
        h_conv1 = tf.nn.relu(preactivated1)
        tf.summary.histogram('pre_activations', preactivated1)
        tf.summary.histogram('activations', h_conv1)
    with tf.name_scope('max_pool'):
        h_pool1 =  tf.nn.max_pool(h_conv1,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], 
                                  padding='SAME')
    # save output of conv layer to TensorBoard - first 16 filters
    with tf.name_scope('Image_output_conv1'):
        image = h_conv1[0:1, :, :, 0:16]
        image = tf.transpose(image, perm=[3,1,2,0])
        tf.summary.image('Image_output_conv1', image)
    # save a visual representation of weights to TensorBoard
with tf.name_scope('Visualise_weights_conv1'):
    # We concatenate the filters into one image of row size 8 images
    W_a = W_conv1                      # i.e. [5, 5, 1, 32]
    W_b = tf.split(W_a, 32, 3)         # i.e. [32, 5, 5, 1, 1]
    rows = []
    for i in range(int(32/8)):
        x1 = i*8
        x2 = (i+1)*8
        row = tf.concat(W_b[x1:x2],0)
        rows.append(row)
    W_c = tf.concat(rows, 1)
    c_shape = W_c.get_shape().as_list()	  
    W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])	  
    tf.summary.image("Visualize_kernels_conv1", W_d, 1024)
#----------------------------------------------------------------------
# Second conv+pool layer
#----------------------------------------------------------------------
with tf.name_scope('conv2'):
    with tf.name_scope('weights'):
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(W_conv2)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv2 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(W_conv2))
            tf.summary.scalar('min', tf.reduce_min(W_conv2))
            tf.summary.histogram('histogram', W_conv2)
    
    with tf.name_scope('biases'):
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(b_conv2)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv2 - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(b_conv2))
            tf.summary.scalar('min', tf.reduce_min(b_conv2))
            tf.summary.histogram('histogram', b_conv2)
    with tf.name_scope('Wx_plus_b'):
        preactivated2 = tf.nn.conv2d(h_pool1, W_conv2,
                                          strides=[1, 1, 1, 1],
                                          padding='SAME') + b_conv2
        h_conv2 = tf.nn.relu(preactivated2)
        tf.summary.histogram('pre_activations', preactivated2)
        tf.summary.histogram('activations', h_conv2)
    with tf.name_scope('max_pool'):
        h_pool2 =  tf.nn.max_pool(h_conv2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], 
                                  padding='SAME')
    # save output of conv layer to TensorBoard - first 16 filters
    with tf.name_scope('Image_output_conv2'):
        image = h_conv2[0:1, :, :, 0:16]
        image = tf.transpose(image, perm=[3,1,2,0])
        tf.summary.image('Image_output_conv2', image)
    # save a visual representation of weights to TensorBoard
with tf.name_scope('Visualise_weights_conv2'):
    # We concatenate the filters into one image of row size 8 images
    W_a = W_conv2
    W_b = tf.split(W_a, 64, 3)
    rows = []
    for i in range(int(64/8)):
        x1 = i*8
        x2 = (i+1)*8
        row = tf.concat(W_b[x1:x2],0)
        rows.append(row)
    W_c = tf.concat(rows, 1)
    c_shape = W_c.get_shape().as_list()	  
    W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])	  
    tf.summary.image("Visualize_kernels_conv2", W_d, 1024)
#----------------------------------------------------------------------
# Fully connected layer
#----------------------------------------------------------------------
with tf.name_scope('Fully_Connected'):
    W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)) #28/(2*2) = 7
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    # Flatten the output of the second pool layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # Dropout
    #keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1.0)
#----------------------------------------------------------------------
# Readout layer
#----------------------------------------------------------------------
with tf.name_scope('Readout_Layer'):
    W_fc2 = tf.Variable(tf.truncated_normal([1024, n_outputs], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[n_outputs]))
# CNN output
with tf.name_scope('Final_matmul'):
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#----------------------------------------------------------------------
# Cross entropy functions
#----------------------------------------------------------------------
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                   logits=y_conv)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)
#----------------------------------------------------------------------
# Optimiser
#----------------------------------------------------------------------
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#----------------------------------------------------------------------
# Accuracy checks
#----------------------------------------------------------------------
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1),
                                      tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# ###START SESSSION AND COMMENCE TRAINING###
#----------------------------------------------------------------------
#----------------------------------------------------------------------
# create session
sess = tf.Session()
# initalise variables
sess.run(tf.global_variables_initializer())

# Merge all the summaries and write them out to "mnist_logs"
merged = tf.summary.merge_all()
output_directory = 'mnist_TB_logs'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    os.makedirs(output_directory + '/train')
    os.makedirs(output_directory + '/test')
# prepare log writers
train_writer = tf.summary.FileWriter(output_directory + '/train', sess.graph)
test_writer = tf.summary.FileWriter(output_directory + '/test')
# prepare checkpoint writer
saver = tf.train.Saver()
#----------------------------------------------------------------------
# Train
#----------------------------------------------------------------------
for i in range(1,training_epochs+1):
    batch_img, batch_lbl = mnist.train.next_batch(batch_size)
    testbatch = mnist.test.next_batch(batch_size)

    # run training step
    sess.run(train_step, feed_dict={x: batch_img,
                                    y_: batch_lbl,
                                    keep_prob: 1.0})

    # output the data into TensorBoard summaries
    if (i)%display_step == 0:
        train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x:batch_img,
                                                                                y_: batch_lbl,
                                                                                keep_prob: 1.0})
        train_writer.add_summary(train_summary, i)
        print("step %d, training accuracy %g"%(i, train_accuracy))

        test_summary, test_accuracy = sess.run([merged, accuracy], feed_dict={x: testbatch[0],
                                                                              y_: testbatch[1],
                                                                              keep_prob: 0.9})
        test_writer.add_summary(test_summary, i)
        print("test accuracy %g"%test_accuracy)
    # output metadata before every full 100 epochs  
    if i % 100 == 0 or i+1 == training_epochs:
        print('Adding run metadata for', i)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x:batch_img, y_: batch_lbl, keep_prob: 1.0},
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
    # save checkpoint at the end
    if i+1 == training_epochs:
        print('\nSaving model at ' + str(i) + 'epochs.')
        saver.save(sess, output_directory + "/model_at_" + str(i) + "_epochs.ckpt",
                   global_step=i)
    


# close writers
train_writer.close()
test_writer.close()

# evaluate accuracy
print('\nEvaluating final accuracy of the model.')
train_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                              y_: mnist.test.labels,
                                              keep_prob: 1.0})
print('Evaluating final accuracy of the model..')
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images,
                                             y_: mnist.train.labels,
                                             keep_prob: 1.0})
print('Evaluating final accuracy of the model...')
val_accuracy  = sess.run(accuracy, feed_dict={x: mnist.validation.images,
                                             y_: mnist.validation.labels,
                                             keep_prob: 1.0})

# print results
print ("\nTraining phase finished")
print ("\tAccuracy for the train set examples      = " , train_accuracy)
print ("\tAccuracy for the test set examples       = " , test_accuracy)
print ("\tAccuracy for the validation set examples = " , val_accuracy)

#print a statement to indicate where to find logs
print('\nRun "tensorboard --logdir=' + output_directory + '" to see results on localhost:6006')