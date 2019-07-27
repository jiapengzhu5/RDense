from Densenet import DenseNet
from cnn_utils import *
from utils import *
from RNA_feature import *
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
growth_k = 24
nb_block = 2 #dense block + Transition Layer
init_learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon
dropout_rate = 0.2
# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4
# Label & batch_size
batch_size = 64
iteration = 938
# batch_size * iteration = data_set_number
test_iteration = 30
total_epochs = 300
fc_size=96

# digit feature
Train_feature_data, Train_feature_lengths, Train_size_f = read_features("E:/RDense/train/RNCMPT00001-digit.txt")
Test_feature_data, Test_feature_lengths, Test_size_f = read_features("E:/RDense/test/RNCMPT00001-digit.txt")

# Sequence
Train_rna_data_seq, Train_rna_lengths_seq, Train_rna_labels ,size1= read_sequence_only("E:/RDense/train/RNCMPT00001.seq-train.txt","E:/RDense/train/RNCMPT00001-struct.txt",41)
Test_rna_data_seq, Test_rna_lengths_seq, Test_rna_labels ,size2= read_sequence_only("E:/RDense/test/RNCMPT00001.seq-test.txt","E:/RDense/test/RNCMPT00001-struct.txt",41)

# Structure
Train_rna_data_str, Train_rna_lengths_str,size3 = read_combined_data("E:/RDense/train/RNCMPT00001.seq-train.txt","E:/RDense/train/RNCMPT00001-struct.txt",41)
Test_rna_data_str, Test_rna_lengths_str,size4= read_combined_data("E:/RDense/test/RNCMPT00001.seq-test.txt","E:/RDense/test/RNCMPT00001-struct.txt",41)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')
is_training = tf.placeholder(tf.bool)

rna_lengths_seq = tf.placeholder(tf.int32, [None])
rna_lengths_str = tf.placeholder(tf.int32,[None])
rna_labels = tf.placeholder(tf.float32, [None])

rna_feature_length = tf.placeholder(tf.int32,[None])
rna_feature = tf.placeholder(tf.float32,[None,8,4])
rna_feature_output = BiRNN(rna_feature,128,rna_feature_length,reuse = None,name="LSTM1")
rna_feature_output = tf.reshape(rna_feature_output,[-1,16,16,1])


rna_data_seq = tf.placeholder(tf.float32, [None, 41,5])
rnn_output_seq=BiRNN(rna_data_seq,128,rna_lengths_seq,reuse=None,name="LSTM2")
rnn_output_seq=tf.reshape(rnn_output_seq,[-1,16,16,1])


rna_data_str = tf.placeholder(tf.float32, [None, 41, 5])
rnn_output_str=BiRNN(rna_data_str,128,rna_lengths_str,reuse = None,name="LSTM3")
rnn_output_str=tf.reshape(rnn_output_str,[-1,16,16,1])

#combine
cnn_input=tf.concat([rnn_output_seq,rnn_output_str,rna_feature_output],3)

# FC
WFC1 = tf.Variable(tf.truncated_normal([fc_size, 128], stddev=0.1))
BFC1 = tf.Variable(tf.zeros(1))
WFC2 = tf.Variable(tf.truncated_normal([128, 1], stddev=0.1))
BFC2 = tf.Variable(tf.zeros(1))

model = DenseNet(x=cnn_input,
     nb_blocks=nb_block,
     filters=growth_k,
     training=is_training,
     # n_class=class_num,
     dropout_rate=dropout_rate)
logits = model.logits
fcl_input = flatten(logits, fc_size)
# self.hidden_layer = nn_layer(self.fcl_input, p.WFC1, p.BFC1, True)
hidden_layer = nn_layer(fcl_input, WFC1, BFC1, True)

# Second fully connected
preds = nn_layer(hidden_layer, WFC2, BFC2, False)
preds = tf.squeeze(preds, axis=1)
print("preds",preds)
# Regularization - ADD MORE ARGS
regularizer = tf.nn.l2_loss(WFC1) + tf.nn.l2_loss(BFC1) + tf.nn.l2_loss(WFC2) + tf.nn.l2_loss(BFC2)

# Loss function
loss_score=tf.reduce_mean(tf.nn.l2_loss(abs((rna_labels - preds)/rna_labels)*100.0/float(64)))
#loss_score = tf.nn.l2_loss(rna_labels - preds)
# cost = loss_score
cost = loss_score + 1.0 / float(64) * 0.001 * regularizer

train_op = tf.train.AdamOptimizer(learning_rate, epsilon).minimize(cost)

# calc accuracy
accuracy = tf.pow(pearson_correlation(preds, rna_labels), 1)


saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)

    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate

    for epoch in range(1, total_epochs + 1):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for step in range(1, iteration + 1):
            if pre_index+batch_size < 60000 :

                batch_x1 = Train_rna_data_seq[pre_index:pre_index+batch_size]
                batch_z1 = Train_rna_lengths_seq[pre_index:pre_index+batch_size]
                batch_y = Train_rna_labels[pre_index:pre_index+batch_size]
                batch_x2 = Train_rna_data_str[pre_index:pre_index+batch_size]
                batch_z2 = Train_rna_lengths_str[pre_index:pre_index+batch_size]
                batch_x3 = Train_feature_data[pre_index:pre_index+batch_size]
                batch_z3 = Train_feature_lengths[pre_index:pre_index+batch_size]
            else :
                batch_x1 = Train_rna_data_seq[pre_index : ]
                batch_y = Train_rna_labels[pre_index : ]
                batch_z1 =Train_rna_lengths_seq[pre_index : ]
                batch_x2 = Train_rna_data_str[pre_index : ]
                batch_z2 =Train_rna_lengths_str[pre_index : ]
                batch_x3 = Train_feature_data[pre_index : ]
                batch_z3 = Train_feature_lengths[pre_index : ]
            # batch_x = data_augmentation(batch_x)

            train_feed_dict = {
                rna_data_seq: batch_x1,
                rna_data_str:batch_x2,
                rna_labels: batch_y,
                rna_lengths_seq:batch_z1,
                rna_lengths_str:batch_z2,
                rna_feature:batch_x3,
                rna_feature_length:batch_z3,
                learning_rate: epoch_learning_rate,
                is_training : True
            }
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                _, batch_loss = sess.run([train_op, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size

            if step == iteration:
                train_loss /= iteration  # average loss
                train_acc /= iteration  # average accuracy

                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])


                def Evaluate(sess):
                    test_acc = 0.0
                    test_loss = 0.0
                    test_pre_index = 0
                    add = 1000

                    for it in range(test_iteration):
                        test_batch_x1 = Test_rna_data_seq[test_pre_index:test_pre_index+batch_size]
                        test_batch_z1 = Test_rna_lengths_seq[test_pre_index:test_pre_index+batch_size]
                        test_batch_x2 = Test_rna_data_str[test_pre_index:test_pre_index+batch_size]
                        test_batch_z2 = Test_rna_lengths_str[test_pre_index:test_pre_index+batch_size]
                        test_batch_y = Test_rna_labels[test_pre_index:test_pre_index+batch_size]
                        test_batch_x3 = Test_feature_data[test_pre_index:test_pre_index+batch_size]
                        test_batch_z3 = Test_feature_lengths[test_pre_index:test_pre_index+batch_size]
                        test_pre_index = test_pre_index + add

                        test_feed_dict = {
                            rna_data_seq: test_batch_x1,
                            rna_data_str:test_batch_x2,
                            rna_labels: test_batch_y,
                            rna_lengths_seq:test_batch_z1,
                            rna_lengths_str:test_batch_z2,
                            rna_feature:test_batch_x3,
                            rna_feature_length:test_batch_z3,
                            learning_rate: epoch_learning_rate,
                            is_training: False
                        }

                        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

                        test_loss += loss_ / 30.0
                        test_acc += acc_ / 30.0

                    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

                    return test_acc, test_loss, summary


                test_acc, test_loss, test_summary = Evaluate(sess)

                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.add_summary(summary=test_summary, global_step=epoch)
                summary_writer.flush()
                line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                    epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
                print(line)

                with open('logs.txt', 'a') as f :
                    f.write(line)
    saver.save(sess=sess, save_path='./model/dense.ckpt')