#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

rm -rf props

for mat in */
do
 # echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   echo -n "${mat::-1} " | tee -a ../props
   cat props | tee -a ../props

   cd ..
done

cd ..
