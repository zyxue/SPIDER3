#!/bin/bash

#######################################
# README!
#
# INPUT_LIST needs to be a file where each line contains the following fields
# <sequence name> <name with file path of PSSM file> <name with file path of HMM profile>
# The file paths need to be either absolute or relative to this script.
#
# the program will append /results/ to the given TMP_DIR.
#
# the NETWORK_DIR must contain each of the networks, as well as the pkl file 
# with normalisation values, split up in folders named i<iteration>-{ss,rest}.
#
#######################################
SECONDS=0

#PDIR=~/work/server/SPIDER3
XDIR=$(dirname $0)
PDIR=$XDIR/../
TMP_DIR='./tmp'
NETWORK_DIR=$PDIR/network_files/
INPUT_LIST=$1
if [ "$INPUT_LIST" == "" ]; then
	echo "usage: $0 list1"; exit 1
fi

for ITER in `seq 0 3`;
do
  ###################
  # SS
  ###################
  echo "doing iteration ${ITER}"
	opt1=""
	if [ $ITER -gt 0 ]; then
	   opt1="--input_ext .i$(( ITER - 1 ))c --input_dir ${TMP_DIR}/"
	fi
# ss
	python $XDIR/spider3_impute.py --saved_network_dir "${NETWORK_DIR}i${ITER}-ss" --input_file_list  ${INPUT_LIST} -o 'ss' -s ${TMP_DIR} --save_ext ".i${ITER}s" $opt1 || exit 1
# ASA THETA TAU PHI PSI HSEa HSEb CN
	python $XDIR/spider3_impute.py --saved_network_dir "${NETWORK_DIR}i${ITER}-rest" --input_file_list  ${INPUT_LIST} -o 'asa' 'ttpp' 'hsea' 'hseb' 'cn' -s ${TMP_DIR} --save_ext ".i${ITER}r" $opt1
# combine
	python $XDIR/combine_outputs_from_file_list.py ${INPUT_LIST} "${TMP_DIR}/" ".i${ITER}s" ".i${ITER}r" ".i${ITER}c" header="spd3 output - iteration ${ITER}" || exit 1
#  echo ${SECONDS}

  ###################
  # convert to readable - optional.
  ###################
  if [ ${ITER} -eq 3 ]  # here we are only doing this for the final iteration.
  then
#  SECONDS=0
  cat ${INPUT_LIST} | awk '{ print $1 }' | xargs -I{} -P4 $XDIR/convert_raw_output_to_readable.py {} ${TMP_DIR}/{}.i${ITER}c ./{}.spd3${ITER} 
#  echo ${SECONDS}
  fi

done
rm -f $TMP_DIR/*.i[0-3][rcs]
rmdir $TMP_DIR/
echo "Time taken - ${SECONDS} seconds"
