#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

rm -rf msmf-min-single
rm -rf msmf-min-double

cd results

for mat in */
do
 # echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   for i in 5 6 7 8 10
   do
      echo -n `cat msmf | awk -v i=$i '{printf("%20d\n", $i)}' | sort -k 1 -n | head -n 1` | tee -a ../../msmf-min-single
      echo -n " " | tee -a ../../msmf-min-single
   done
   echo "" | tee -a ../../msmf-min-single

   for i in 5 6 7 9 11
   do
      echo -n `cat msmf | awk -v i=$i '{printf("%20d\n", $i)}' | sort -k 1 -n | head -n 1` | tee -a ../../msmf-min-double
      echo -n " " | tee -a ../../msmf-min-double
   done
   echo "" | tee -a ../../msmf-min-double

   cd ..
done

cd ..
