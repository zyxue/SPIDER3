#!/bin/bash

NR90=/home/pub/blast/NR90/nr90
HHBLITS=/export/home/s2878100/local/hhsuite/bin/hhblits
HHDB=/export/home/s2878100/aspen/pub/uniprot20_2013_03/uniprot20_2013_03
psiblast=/export/home/s2878100/aspen/software/blast-2.2.30+/bin/psiblast
#

PDIR=$(dirname $0)
xdir=$PDIR/scripts
ncpu=$OMP_NUM_THREADS
if [ "$ncpu" == "" ]; then ncpu=16; fi

if [ $# -lt 1 ]; then echo "usage: $0 *.seq"; exit 1; fi

rm -f tlist
for seq1 in $*; do
	pro1=$(basename $(basename $seq1 .seq) .pssm)
	[ -f $pro1.spd33 ] && continue
	[ -f $pro1.pssm ] || $psiblast/psiblast -db $NR90 -num_iterations 3 -num_alignments 1 -num_threads $ncpu -query $pro1.seq -out  ./$pro1.bla -out_ascii_pssm ./$pro1.pssm #-out_pssm ./$pro1.chk
	[ -f $pro1.hhm ] || $HHBLITS -i $seq1 -ohhm $pro1.hhm -d $HHDB -v0 -maxres 40000 -cpu $ncpu -Z 0
	[ -f $pro1.hhm -a -f $pro1.pssm ] && echo $seq1 $pro1.pssm $pro1.hhm >> tlist
done
if [  ! -s tlist ]; then
	echo "empty file: tlist has no files having pssm/hhm"
	exit 1
fi
#
module load gcc/4.9.3
module load glibc/2.14
$xdir/impute_script.sh tlist
