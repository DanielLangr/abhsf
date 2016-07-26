#!/bin/bash

ABHSF_ROOT="${HOME}/projects/abhsf"

cyan='\033[0;36m'
reset='\033[0m'

cd results

for mat in */
do
   echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat
   $ABHSF_ROOT/block_statistics/mmf
   cd ..
done

cd ..
