
####################################################  
import tensorflow as tf
import numpy as np
from dynamic_brnn import dynamic_bidirectional_rnn


####################################################
# DEFINE SOME VARIABLES/PARAMETERS
####################################################
 
def fully_connected(input, weight_shape, bias_shape):
  # shape should be a list = [input_size, output_size]
  weights = tf.get_variable("weights", weight_shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.025))
  bias = tf.get_variable("bias", bias_shape, initializer=tf.constant_initializer(0.01))
  
  return tf.matmul(input, weights) + bias
   
def num_batches(num_vec, batch_size):
  incomplete_batch = 1 if np.mod(num_vec, batch_size) else 0
  return num_vec/batch_size+incomplete_batch
  


####################################################
# DEFINE NETWORK CLASSES
####################################################

class output_layer:
  # this layer's input is the output of a fully_connected_layer_with_dropout
  # the input shape will be 2D tensor of shape [SUM(sequences in batch lengths), output_size]
  # the output should be [SUM(sequences in batch lengths), output_size]
  def __init__(self, layer_input, input_size, output_size):
    self.output = fully_connected(layer_input, [input_size, output_size], [output_size])
    
   
  
class fully_connected_layer_with_dropout:
  # this layer's input is the output of a rnn_output_to_full_connected_reshape layer.
  # the input shape will be a 2D tensor of [batch_size*max_seq_len, output_size]
  # the output should be [batch_size*max_seq_len, output_size]
  def __init__(self, layer_input, input_size, layer_size, keep_prob):
    self.layer_input = layer_input
    self.keep_prob = keep_prob
    self.layer_size = layer_size
    self.input_size = input_size
    
    self.output = tf.nn.relu(fully_connected(self.layer_input, [self.input_size, self.layer_size], [self.layer_size]))
    # self.output at this point is: 2D tensor of [batch_size*max_seq_len, output_size] 
    
    self.output = tf.nn.dropout(self.output, self.keep_prob)
    # self.output at this point is: 2D tensor of [batch_size*max_seq_len, output_size] 
    
class rnn_output_to_fully_connected_reshape:
  # this layer's input is the output of a (bi)rnn layer.
  # the input shape will be a 3D tensor of [batch_size, seq_len (n_steps), 2*rnn_layer_size (2*input_size)]
  # the output should be [batch_size*max_seq_len, output_size]
  # the output should be [SUM(seq_lengths), output_size]
  
  def __init__(self, layer_input, bool_length_mask, prev_layer_size):
    self.layer_size = prev_layer_size
    self.output = tf.boolean_mask(layer_input, bool_length_mask)

    
class brnn_layer:
  # input to this layer will be a list of n_steps length, with each element being a 2D
  # tensor of shape [batch_size, n_input]
  # output from this layer will a list of n_steps length, with each element being a 2D
  # tensor of shape [batch_size, 2*layer_size]
  def __init__ (self, ph_layer_input, ph_seq_lengths, n_input, layer_size, scope="RNN"):
    self.ph_layer_input = ph_layer_input
    self.ph_seq_lengths = ph_seq_lengths
    self.n_input = n_input
    self.layer_size = layer_size
    
    # define basic lstm cells
    self.lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layer_size,state_is_tuple=True)
    self.lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.layer_size,state_is_tuple=True)
    
    self.output, output_state_fw, output_state_bw = dynamic_bidirectional_rnn(self.lstm_fw_cell,
                                            self.lstm_bw_cell, self.ph_layer_input,
                                            sequence_length=self.ph_seq_lengths,
                                            dtype="float", scope=scope)
                                            
                                            
                                            
def bioinf_output_loss(output_type, true, pred, mask):
  # this functions takes a single output_type and returns the loss for that 
  # output type, given true and predicted values.
  # true and pred are tensors of shape [num_'frames', num_classes]
  # for output types such as SS, where we are doing classifiction, num_classes = 1 and we use
  # cross entropy loss.
  #
  # the true and pred being passed in should already be masked if they need to be.
  
  output_type = output_type.upper()
  
  if output_type == 'SS':
    # apply the mask here.
    masked_true = tf.to_int32(tf.boolean_mask(true, mask))
    masked_pred = tf.boolean_mask(pred, tf.reshape(mask, [-1]))    
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(masked_pred, masked_true))
  elif output_type == 'ASA' or output_type == 'HSEA' or output_type == 'HSEB' or output_type == 'CN' or output_type == 'CN13' or output_type == 'THETA' or output_type == 'TAU' or output_type == 'PHI' or output_type == 'PSI' or output_type == 'TT' or output_type == 'PP':
    masked_true = tf.boolean_mask(true, mask)
    masked_pred = tf.sigmoid(tf.boolean_mask(pred, mask))
    loss = tf.nn.l2_loss(masked_true - masked_pred)
  elif output_type == 'TTPP':
    masked_true = tf.boolean_mask(true, mask)
    masked_pred = tf.sigmoid(tf.boolean_mask(pred, mask))
#    masked_pred = tf.tanh(tf.boolean_mask(pred, mask))
    loss = tf.nn.l2_loss(masked_true - masked_pred)
    
  return loss
    
    
def bioinf_output_nonlinearity(output_type, pred):
  # this function applies the nonlinear activation functions for the different output types.
  
  output_type = output_type.upper()
  if output_type == 'SS':
    non_linear_output = tf.nn.softmax(pred)
  elif output_type == 'ASA' or output_type == 'HSEA' or output_type == 'HSEB' or output_type == 'CN' or output_type == 'CN13' or output_type == 'THETA' or output_type == 'TAU' or output_type == 'PHI' or output_type == 'PSI' or output_type == 'TT' or output_type == 'PP':
    non_linear_output = tf.sigmoid(pred)
  elif output_type == 'TTPP':
#    non_linear_output = tf.tanh(pred)
    non_linear_output = tf.sigmoid(pred)
    
    
  return non_linear_output

class brnn_network:
  def __init__(self, layer_sizes, output_types, output_index_true, output_index_pred,
                ph_network_input, n_input, 
#                n_steps,
                ph_seq_lengths, ph_seq_length_mask, ph_bool_length_mask, ph_network_output, ph_network_output_mask, ph_network_output_mask_encoded, n_classes, ph_keep_prob):   
    # network variables
    self.layer_sizes = layer_sizes
    self.output_types = output_types  # output_type is a list of stings for each output type.
    self.output_index_true = output_index_true # this is a list of lists of start and stop index for the true labels.
    self.output_index_pred = output_index_pred # this is a list of lists of start and stop index for the predicted labels.
    # note that the true labels will only contain an int for class labels (ie. the int will represent the class), while the
    # predicted labels will be using a one hot representation for class labels. This will cause the indexs to be different in the two cases.
    # the true and predicted labels will be the same shape for regression tasks.
    
    self.ph_seq_lengths = ph_seq_lengths
    self.n_input = n_input
    self.n_classes = n_classes
    self.ph_network_input = ph_network_input  # network input is the input data
    self.ph_network_output = ph_network_output # network output is the true labels
    self.ph_network_output_mask = ph_network_output_mask  # network output mask is a set of masks to remove undefined data points (ie X for SS class, or 180 for ASA values etc.)
    self.ph_network_output_mask_encoded = ph_network_output_mask_encoded  # network output mask of the same shape as the network's predictions
    self.ph_network_output_bool_mask = ph_network_output_mask>0  # network output mask is a set of masks to remove undefined data points (ie X for SS class, or 180 for ASA values etc.)
    self.ph_seq_len_mask = ph_seq_length_mask
    self.ph_bool_len_mask = ph_bool_length_mask
    self.ph_keep_prob = ph_keep_prob
    
    
    # network layers
    self.layer = []

    # reshape for LSTM inputs    
    self.ph_lstm_input = self.ph_network_input
    
    # lstm layers
    self.layer.append(brnn_layer(self.ph_lstm_input, 
                                 self.ph_seq_lengths, 
                                 self.n_input, 
#                                 self.n_steps,
                                 self.layer_sizes[0][0],
                                 scope="RNN1"
                                 ))
                                 
#    self.lstm_1_output = tf.pack(self.layer[-1].output)
    self.lstm_1_output = tf.nn.dropout(self.layer[-1].output, self.ph_keep_prob)
    
    self.layer.append(brnn_layer(self.lstm_1_output,
                                 self.ph_seq_lengths, 
                                 2*self.layer[-1].layer_size,
#                                 self.n_steps,
                                 self.layer_sizes[0][1],
                                 scope="RNN2"
                                 ))
                                 
    # reshape LSTM outputs    
    self.lstm_output = tf.nn.dropout(self.layer[-1].output, self.ph_keep_prob)
    
    # reshape layer
    self.layer.append(rnn_output_to_fully_connected_reshape(self.lstm_output, self.ph_bool_len_mask,
                                                            2*self.layer[-1].layer_size))
    
    # fully connected layers
    for fc_layer_num, n_hidden in enumerate(self.layer_sizes[1]):
      with tf.variable_scope("fully_connected"+str(fc_layer_num)):
        self.layer.append(fully_connected_layer_with_dropout(self.layer[-1].output,
                                                             self.layer[-1].layer_size,
                                                             n_hidden,
                                                             self.ph_keep_prob))
                            
    # output layer
    with tf.variable_scope("output_layer"):
#      self.layer.append(fully_connected(self.layer[-1].output, [self.layer[-1].layer_size, self.n_classes], [self.n_classes]))
      self.layer.append(output_layer(self.layer[-1].output, self.layer[-1].layer_size, self.n_classes))

    self.pred = self.layer[-1].output
    self.linear_output = self.pred
    
    self.masked_pred = tf.mul(self.ph_network_output_mask_encoded, self.pred)
    self.masked_network_output = tf.mul(self.ph_network_output_mask, self.ph_network_output)
    
    temp_loss = []
    temp_non_linear_output = []
    for ind, output_type in enumerate(output_types):
          
      temp_loss.append(bioinf_output_loss(output_type, self.ph_network_output[:, self.output_index_true[ind][0]:self.output_index_true[ind][1]],
                                                       self.pred[:,self.output_index_pred[ind][0]:self.output_index_pred[ind][1]],
                                                       self.ph_network_output_bool_mask[:, self.output_index_true[ind][0]:self.output_index_true[ind][1]] ) )
                         
      temp_non_linear_output.append(bioinf_output_nonlinearity(output_type, self.linear_output[:, self.output_index_pred[ind][0]:self.output_index_pred[ind][1]]))
    
    self.non_linear_output = tf.concat(1, temp_non_linear_output) 
    self.loss = tf.add_n(temp_loss)

    
  def get_predictions(self, input_feat, seq_len, batch_size=500, keep_prob=1.0):
    # this function will do a forward pass of the network and will return a set of 
    # predictions for each of the inputs.
    # the input to this function should be all of the input data you want predictions for
    # as well as the sequence masks that go along with the input data.
    
    
    for i in xrange(0, num_batches(len(input_feat), batch_size)):
      batch_ind = range(i*batch_size, np.minimum((i+1)*batch_size, len(input_feat)))
      batch_seq_lengths = [ seq_len[ind] for ind in batch_ind ]
      batch_max_length = max(batch_seq_lengths)
      batch_feat = np.array( [ np.concatenate((np.array(tmp), np.zeros((batch_max_length - tmp.shape[0], len(input_feat[0][0]))))) for tmp in [ input_feat[ind] for ind in batch_ind ] ] )
      batch_seq_len_mask = np.array( [ np.concatenate((np.ones(tmp), np.zeros(batch_max_length - tmp))) for tmp in batch_seq_lengths ] )

      feed_dict={self.ph_network_input: batch_feat,
                 self.ph_keep_prob: keep_prob,
                 self.ph_seq_lengths: batch_seq_lengths,
                 self.ph_bool_len_mask: batch_seq_len_mask.astype(bool)}
      temp = self.non_linear_output.eval(feed_dict)
      
      # here would be a good place to convert from shape [SUM(batch lengths), # classes] to
      # a more useful shape [sequence length, # classes]
      
      if i == 0:
        np_output = temp
      else:
        np_output = np.concatenate((np_output, temp))    
    
    return np_output
    
   
