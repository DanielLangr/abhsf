#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

rm -rf rows-nnz-stdev

for mat in */
do
 # echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   echo -n "${mat::-1} " | tee -a ../rows-nnz-stdev
   ../../../rows-nnz-stdev | tee -a ../rows-nnz-stdev

   cd ..
done

cd ..
