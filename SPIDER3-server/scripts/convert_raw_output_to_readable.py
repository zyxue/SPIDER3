#!/usr/bin/env python

def rdfasta(fn):
	name = seq1 = ''
	for i,x in enumerate(open(fn)):
		if i==0 and x[0]=='>': name = x[1:].strip()
		elif x[0] == '>': break
		else: seq1 += x.strip().upper()
	return name, seq1
#
def convert_raw_file(seq_filename, input_filename, output_filename):
  import numpy as np

  name, seq1 = rdfasta(seq_filename)
  raw_data = np.loadtxt(input_filename)

  assert len(seq1) == len(raw_data)
  rnam1_std = "ACDEFGHIKLMNPQRSTVWY"
  ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
			185, 160, 145, 180, 225, 115, 140, 155, 255, 230)
  dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
  ASA0 = np.asarray([dict_rnam1_ASA[x] for x in seq1])

  ss_order = ['C','H','E']
  ss_ind = np.argmax(raw_data[:,0:3], axis=1)
  pred_ss = np.array([ss_order[i] for i in ss_ind])

  pred_asa = raw_data[:,3] * ASA0

  raw_ttpp = raw_data[:,4:12] * 2 - 1;
  pred_theta = np.rad2deg(np.arctan2(raw_ttpp[:,0], raw_ttpp[:,4]))
  pred_tau = np.rad2deg(np.arctan2(raw_ttpp[:,1], raw_ttpp[:,5]))
  pred_phi = np.rad2deg(np.arctan2(raw_ttpp[:,2], raw_ttpp[:,6]))
  pred_psi = np.rad2deg(np.arctan2(raw_ttpp[:,3], raw_ttpp[:,7]))

  pred_hsea_up = raw_data[:,12] * 50.
  pred_hsea_down = raw_data[:,13] * 65.
  pred_hseb_up = raw_data[:,14] * 50.
  pred_hseb_down = raw_data[:,15] * 65.

  pred_cn = raw_data[:,16] * 85.

  readable_data = np.zeros(pred_ss.size,
          dtype=[
                 ('index', int),
                 ('pred_seq', 'S1'),
                 ('pred_ss', 'S1'),
                 ('pred_asa', float),
                 ('pred_phi', float),
                 ('pred_psi', float),
                 ('pred_theta', float),
                 ('pred_tau', float),
                 ('pred_hseau', float),
                 ('pred_hsead', float),
                 ('pred_pc', float),
                 ('pred_ph', float),
                 ('pred_pe', float) ])

  readable_data['index'] = np.arange(len(pred_ss)) + 1
  readable_data['pred_ss'] = pred_ss
  readable_data['pred_seq'] = np.array(list(seq1))
  readable_data['pred_asa'] = pred_asa
  readable_data['pred_phi'] = pred_phi
  readable_data['pred_psi'] = pred_psi
  readable_data['pred_theta'] = pred_theta
  readable_data['pred_tau'] = pred_tau
  readable_data['pred_hseau'] = pred_hsea_up
  readable_data['pred_hsead'] = pred_hsea_down
  readable_data['pred_pc'] = raw_data[:,0]
  readable_data['pred_ph'] = raw_data[:,1]
  readable_data['pred_pe'] = raw_data[:,2]
#  readable_data['pred_hsebu'] = pred_hseb_up
#  readable_data['pred_hsebd'] = pred_hseb_down
#  readable_data['pred_cn'] = pred_cn

  np.savetxt(output_filename, readable_data, fmt="%-3d %s %s %5.1f %6.1f %6.1f %6.1f %6.1f %4.1f %4.1f %5.3f %5.3f %5.3f", header='SEQ SS ASA Phi Psi Theta(i-1=>i+1) Tau(i-2=>i+2) HSE_alpha_up HSE_alpha_down P(C) P(H) P(E)')


if __name__ == "__main__":
  import sys

  if len(sys.argv) == 4:
    seq_filename, input_filename, output_filename = sys.argv[1:4]
  else:
    print 'invalid number of inputs'
    exit

  convert_raw_file(seq_filename, input_filename, output_filename)
