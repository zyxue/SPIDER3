#!/usr/bin/env python
import sys, os
pdir = os.path.dirname(sys.argv[0])
sys.path.insert(0, os.path.join(pdir, 'source'))

####################################################
# DEFINE SOME FUNCTIONS
####################################################
def brnn_impute(directory_to_saved_network, input_types, output_types, network_size=[[256,256],[1024,512]], output_dir=None, print_results=False, input_file_dir=None, save_file_ext='.spd3', input_file_ext='.spd3'):
  # input_types is either the list of input types - same as brnn.py
  #   or it can be a filename - for example the casp data.
  #   if it is the filename we are currently making a heap of assumptions about what type of data it is.
  # output_types does the same thing. either the same as previously (for mat files), a list of the outputs for casp, or <n_classes> for no accuracy testing.
  import numpy as np
  import tensorflow as tf
  import load_bioinf_data as load_data
  import brnn_network_class as brnn_network
  import sys, time, random
  import scipy.io as sp
  import misc_functions as misc
  import os
  import pickle


  # this function is the wrapper function for the main body of code in this file.
  # this function is what is called by any script

  ####################################################
  # DEFINE SOME VARIABLES/PARAMETERS
  ####################################################
  scope_str = 'full'

  if print_results is True:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

  ####################################################
  # LOAD THE DATA
  ####################################################

  # LOAD THE TRAINING DATA NORMALISATION STATS!!
  fp = open(directory_to_saved_network+'/data_stats_' + scope_str + '.pkl','r')
  feat_mean, feat_var = pickle.load(fp)
  fp.close()

  test_seq_names, test_feat, feature_length = load_data.load_spd3_input_wrapper(input_types, feat_mean, feat_var, input_file_dir=input_file_dir, input_file_ext=input_file_ext)

  true_label_ind, pred_label_ind, n_classes = load_data.get_outputs_list_stub(output_types)

  test_lengths = [ len(tmp) for tmp in test_feat ];

  # Network Parameters
  n_input = feature_length # this is the size of the features, ie 20 for PSSM

  # tf Graph input
  x = tf.placeholder("float", [None, None, n_input])
  ph_seq_len = tf.placeholder(tf.int64, [None])
  ph_seq_len_mask = tf.placeholder(tf.int32, [None, None])
  ph_bool_len_mask = tf.placeholder(tf.bool, [None, None])
  ph_keep_prob = tf.placeholder("float")

  # I don't know if I need these any more?
  y = tf.placeholder("float", [None, true_label_ind[-1][-1]])
  ph_output_mask = tf.placeholder("float", [None, true_label_ind[-1][-1]])
  ph_output_mask_encoded = tf.placeholder("float", [None, pred_label_ind[-1][-1]])



  ####################################################
  # DO ACTUAL TRAINING/TESTING
  ####################################################
  with tf.Session() as sess:
    with tf.variable_scope(scope_str):

      # init the network
      network = brnn_network.brnn_network(network_size, output_types, true_label_ind, pred_label_ind,
                                          x, n_input, ph_seq_len, ph_seq_len_mask, ph_bool_len_mask,
                                          y, ph_output_mask,ph_output_mask_encoded, n_classes, ph_keep_prob)

      # start the saver
      saver = tf.train.Saver()

      # reload the best network.
      saver.restore(sess, directory_to_saved_network + '/network_best_'+scope_str)


      # get our network outputs for the best network.
      test_network_output = network.get_predictions(test_feat, test_lengths, batch_size=100)

      # save the outputs from the best network.
      if print_results is True:
        #print "Saving the best outputs of the test set to files in "+directory_to_save_files+'results'

        if not os.path.isdir(output_dir):
          os.makedirs(output_dir)

        # save the results to output files.
        misc.save_predictions_to_file(test_network_output, test_seq_names, test_lengths, save_dir=output_dir, file_ext=save_file_ext, header='%s' % ', '.join(map(str, output_types)))



if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--saved_network_dir',
                      dest="directory_to_saved_networks",
                      help="directory where the network and normalisation values are saved")
  parser.add_argument('-i', '--input_file_list',
                      dest="input_file_list",
                      help="list of inputs. each line should contain <seq name> <pssm file> <hhm file>.")
  parser.add_argument('-o','--output_types',
                      nargs='+',
                      dest="output_types",
                      default=['ss'],
                      help='output types for the network.')
  parser.add_argument('-s', '--output_save_directory',
                      dest="directory_to_save_outputs",
                      default='./',
                      help='directory to save all files to.')
  parser.add_argument('--save_ext',
                      dest="save_ext",
                      help="file extension for the output files")
  parser.add_argument('--input_ext',
                      dest="input_ext",
                      help="file extension of the previous outputs (being used as inputs for this iteration)",
                      default=None)
  parser.add_argument('--input_dir',
                      dest="input_dir",
                      help="directory of input files",
                      default=None)

  args = parser.parse_args()

## Here is argparse style inputs.
  brnn_impute(args.directory_to_saved_networks, args.input_file_list, args.output_types, print_results=True, output_dir=args.directory_to_save_outputs, save_file_ext=args.save_ext, input_file_dir=args.input_dir, input_file_ext=args.input_ext)
