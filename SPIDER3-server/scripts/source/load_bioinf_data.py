import numpy as np
import scipy.io as sio


###############################################################################  N
def load_spd3_input_wrapper(filename_list, feat_mean, feat_var, input_file_dir=None, input_file_ext='.spd3'):
  # this function will take a file as input where each line in the file has 3 fields
  # SEQ_NAME SEQ_PSSM_FILE_PATH SEQ_HMM_FILE_PATH
  # we get the primary sequence from the pssm file.
  # we want to output pssm + hmm + phys7.
  
  # define the dictionary with the phys properties for each AA
  phys_dic = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
              'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
              'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
              'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
              'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
              'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
              'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
              'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
              'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
              'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
              'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
              'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
              'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
              'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
              'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
              'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
              'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
              'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
              'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
              'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476],
              'X': [ 0.0771,-0.1536, -0.0620, -0.0762, -0.1451,  0.0497, -0.0398],
              'Z': [ 0.0771,-0.1536, -0.0620, -0.0762, -0.1451,  0.0497, -0.0398]}
 
  with open(filename_list) as fp:
    lines = fp.readlines()

  all_seq_names = []
  all_seq_data = []

  for line in lines:

    temp_line = line.split()

    if line[0]=='#' or len(temp_line) not in (1,3):
      print 'skipped line', line,
      continue
    elif len(temp_line) == 1:
      x1 = temp_line[0]
      if x1.endswith('.seq'): x1 = x1[:-4]
      temp_line += [x1+'.pssm', x1+'.hhm']

    all_seq_names.append(temp_line[0])
    seq_aa, pssm = read_pssm(temp_line[1])
    seq_phys = np.array( [ phys_dic[i] for i in seq_aa ] )
    seq_pssm = np.array(pssm)
    seq_hmm  = np.array(read_hmm(temp_line[2])[1])
    if input_file_dir is not None:
      seq_file_input = load_spd3_file(input_file_dir + temp_line[0] + input_file_ext)
      seq_data = np.concatenate([seq_pssm, seq_phys, seq_hmm, seq_file_input], axis=1)
    else:
      seq_data = np.concatenate([seq_pssm, seq_phys, seq_hmm], axis=1)



    all_seq_data.append(seq_data)

  feature_length = len(all_seq_data[0][0])
  # normalise the data
  all_seq_data_norm = do_mv_normalisation(all_seq_data, normalisation_mask=None, input_mean=feat_mean, input_var=feat_var)[0]
  
  return all_seq_names, all_seq_data_norm, feature_length
 

############################################################################### N
def read_pssm(pssm_file):
  # this function reads the pssm file given as input, and returns a LEN x 20 matrix (list) of pssm values.

  # index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
  idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)
  
  # open the two files, read in their data and then close them
  fp = open(pssm_file, 'r')
  lines = fp.readlines()
  fp.close()

  # declare the empty dictionary with each of the entries
  aa = []
  pssm = []
  
  # iterate over the pssm file and get the needed information out
  for line in lines:
    split_line = line.split()
    # valid lines should have 32 points of data.
    # any line starting with a # is ignored
    if (len(split_line) == 44) and (split_line[0] != '#'):
      aa_temp = split_line[1]
      aa.append(aa_temp)
      pssm_temp = [-float(i) for i in split_line[2:22]]
      pssm.append([pssm_temp[k] for k in idx_res])
  
  return aa, pssm


############################################################################### N
def read_hmm(hhm_file):
  f = open(hhm_file)
  line=f.readline()
  while line[0]!='#':
      line=f.readline()
  f.readline()
  f.readline()
  f.readline()
  f.readline()
  seq = []
  extras = np.zeros([0,10])
  prob = np.zeros([0,20])
  line = f.readline()
  while line[0:2]!='//':
      lineinfo = line.split()
      seq.append(lineinfo[0])  
      probs_ = [2**(-float(lineinfo[i])/1000) if lineinfo[i]!='*' else 0. for i in range(2,22)]
      prob = np.concatenate((prob,np.matrix(probs_)),axis=0)
      
      line = f.readline()
      lineinfo = line.split()
      extras_ = [2**(-float(lineinfo[i])/1000) if lineinfo[i]!='*' else 0. for i in range(0,10)]
      extras = np.concatenate((extras,np.matrix(extras_)),axis=0)
      
      line = f.readline()
      assert len(line.strip())==0
      
      line = f.readline()
  #return (''.join(seq),prob,extras)
  return (seq,np.concatenate((prob,extras),axis=1))


############################################################################### N
def do_mv_normalisation(data, normalisation_mask=None, input_mean=None, input_var=None):
  # does 0 mean unit variance normalisation
    
  if normalisation_mask is None:
    normalisation_mask = np.ones(data[0].shape[1]) # THIS MAY NOT WORK FOR NPARRAY?
  
  if input_mean is None:
    input_mean, input_var = get_mean_variance(data)

  # do the masking
  input_mean[normalisation_mask==0] = 0
  input_var[normalisation_mask==0] = 1
    
  # do the normalisation
  if type(data) is np.array:
    normalised_data = (data - input_mean) / np.sqrt(input_var)
  if type(data) is list:
    normalised_data = [(tmp - input_mean) / np.sqrt(input_var) for tmp in data]
  
   
  return normalised_data, input_mean, input_var


  
############################################################################## N
def load_spd3_file(filename):
  # this function should load one file.
  
  read_data = np.loadtxt(filename)
  
  return read_data



################################################################################ N 
def process_label_for_one_seq_one_type_stub(output_type=None):
  
  output_type = output_type.upper()
  
  if output_type == 'SS':
    true_label_size = 1
    pred_label_size = 3
  elif output_type == 'ASA':
    true_label_size = 1
    pred_label_size = 1
  elif output_type == 'TTPP':  
    true_label_size = 8
    pred_label_size = 8
  elif output_type == 'THETA':  
    true_label_size = 2
    pred_label_size = 2
  elif output_type == 'TAU':  
    true_label_size = 2
    pred_label_size = 2
  elif output_type == 'TT':  
    true_label_size = 4
    pred_label_size = 4
  elif output_type == 'PHI':  
    true_label_size = 2
    pred_label_size = 2
  elif output_type == 'PSI':  
    true_label_size = 2
    pred_label_size = 2
  elif output_type == 'PP':  
    true_label_size = 4
    pred_label_size = 4
  elif output_type == 'HSEA':
    true_label_size = 2
    pred_label_size = 2
  elif output_type == 'HSEB':
    true_label_size = 2
    pred_label_size = 2
  elif output_type == 'CN' or output_type == 'CN13':
    true_label_size = 1
    pred_label_size = 1
  else:    
    label = None
    print "ERROR GETTING OUTPUT LABELS ", output_type, " IS NOT VALID"

  return true_label_size, pred_label_size

############################################################################### N
def process_labels_for_one_seq_all_types_stub(output_types=None):  
  true_label_size, pred_label_size = process_label_for_one_seq_one_type_stub(output_types[0])
  list_of_true_label_sizes = [true_label_size]
  list_of_pred_label_sizes = [pred_label_size]
  
  for _type in output_types[1:]:
    t_true_label_size, t_pred_label_size = process_label_for_one_seq_one_type_stub(_type)
    list_of_true_label_sizes = list_of_true_label_sizes + [t_true_label_size]   
    list_of_pred_label_sizes = list_of_pred_label_sizes + [t_pred_label_size]  
  
  return list_of_true_label_sizes, list_of_pred_label_sizes


############################################################################### N
def get_outputs_list_stub(output_types):

  list_of_true_label_sizes, list_of_pred_label_sizes = process_labels_for_one_seq_all_types_stub(output_types=output_types)
    
  list_of_true_label_sizes = [0] + list_of_true_label_sizes
  true_label_ind = [ [sum(list_of_true_label_sizes[0:i])] + [sum(list_of_true_label_sizes[0:i+1])] for i in range(1, len(list_of_true_label_sizes)) ]

  list_of_pred_label_sizes = [0] + list_of_pred_label_sizes  
  pred_label_ind = [ [sum(list_of_pred_label_sizes[0:i])] + [sum(list_of_pred_label_sizes[0:i+1])] for i in range(1, len(list_of_pred_label_sizes)) ]

  n_classes = sum(list_of_pred_label_sizes)  
  
  return true_label_ind, pred_label_ind, n_classes
 
  
  
  
