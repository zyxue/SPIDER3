import numpy as np
  
def save_predictions_to_file(network_output, seq_names, seq_lengths, save_dir='.', file_ext='.spd3', header=''):
  # this function should take all of the network's predictions and save them to
  # individual files (in the save_dir directory, using file_ext as the file
  # extension).
  # header will be printed as a header to the file. This may be (for example)
  # the output types.
  
  # split network_output (which will be a large numpy array) into a list.
  # each element of the list should be a single sequence.
  temp_seq_lengths = [0,]+seq_lengths
  network_output_list = []
  for ind in range(len(seq_lengths)):
    network_output_list.append(network_output[sum(temp_seq_lengths[0:ind+1]):sum(temp_seq_lengths[0:ind+2]),:])
    
  # save each of those sequence predictions to a file.
  for ind, pred in enumerate(network_output_list):
    # write the prediction to a file
    str_name = save_dir+'/'+seq_names[ind]+file_ext
    np.savetxt(str_name, pred, header=header)

