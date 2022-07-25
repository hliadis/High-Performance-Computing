#!/bin/sh
#Executing app 12 times.
CUR_DIR="`pwd`"
EXEC_DIR="${CUR_DIR}/seq_kmeans/"
IMAGE_DIR="${EXEC_DIR}/Image_data/"
NUM_CLUSTERS=$2
TIMES=2
TOTAL_THREADS=2


for j in 1 2 3 4 5
do
TOTAL_THREADS=$((TOTAL_THREADS * TIMES)) 
 export OMP_NUM_THREADS=${TOTAL_THREADS} 
for i in 1 2 3 4 5 6 7 8 9 10 11 12
  do 
    bash -c "${EXEC_DIR}/seq_main -q -o -b -n ${NUM_CLUSTERS} -i ${IMAGE_DIR}texture17695.bin"
 done > "${EXEC_DIR}test/out_$1_${TOTAL_THREADS}"

flag=0
compare="Computation"

if [ -e ${EXEC_DIR}test/time_$1_${TOTAL_THREADS} ]
then
rm ${EXEC_DIR}test/time_$1_${TOTAL_THREADS}
fi
input="${EXEC_DIR}test/out_$1_${TOTAL_THREADS}"
while IFS=' ' read -r F1 F2 F3 F4 F5 
do
  if [ "$F1" = "$compare" ]
  
  then
     echo $F4 >> "${EXEC_DIR}test/time_$1_${TOTAL_THREADS}"
  fi
done < "$input"
rm "${EXEC_DIR}test/out_$1_${TOTAL_THREADS}"
done
export OMP_NUM_THREADS=DEFAULT
