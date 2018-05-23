
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow import pack, unpack, reshape, transpose
import tensorflow as tf

#from tensorflow.models.rnn import rnn as tf_rnn

###################################################
# Changed for TF version 0.8
#


def dynamic_bidirectional_rnn(cell_fw, cell_bw, inputs,
                        initial_state_fw=None, initial_state_bw=None,
                        dtype=None, sequence_length=None, scope=None):
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      [batch_size x cell.state_size].
    initial_state_bw: (optional) Same as for initial_state_fw.
    dtype: (optional) The data type for the initial state.  Required if either
      of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size [batch_size],
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs is a length T list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs
      output_state_fw is the final state of the forward rnn
      output_state_bw is the final state of the backward rnn

  Raises:
    TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, rnn_cell.RNNCell):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not isinstance(cell_bw, rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")
  if inputs is None:
    raise ValueError("inputs must not be empty")
    

  name = scope or "BiRNN"
  # Forward direction
  with vs.variable_scope(name + "_FW") as fw_scope:
    output_fw, output_state_fw = tf.nn.dynamic_rnn(cell_fw, inputs,sequence_length=sequence_length,
                                 initial_state=initial_state_fw, dtype=dtype,scope=fw_scope, time_major=False)
  
  # Backward direction
  with vs.variable_scope(name + "_BW") as bw_scope:
    tmp, output_state_bw = tf.nn.dynamic_rnn(cell_bw, tf.reverse_sequence(inputs, sequence_length,1,batch_dim=0), sequence_length=sequence_length, 
                                  initial_state=initial_state_bw, dtype=dtype, scope=bw_scope, time_major=False)
  output_bw = tf.reverse_sequence(tmp, sequence_length,1,batch_dim=0)
  
  # Concat each of the forward/backward outputs
  outputs = array_ops.concat(2, [output_fw, output_bw])

  return (outputs, output_state_fw, output_state_bw)
  
 
  
  
