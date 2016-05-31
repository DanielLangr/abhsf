#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

rm -rf msmf-min-single
rm -rf msmf-min-double

for mat in */
do
 # echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   echo -n "${mat::-1} " | tee -a ../msmf-min-single
   echo -n `cat props | awk '{printf "%14d", $3}'` | tee -a ../msmf-min-single 
   echo -n " " | tee -a ../msmf-min-single
   for i in 5 6 7 8 10
   do
      echo -n `cat msmf | awk -v i=$i '{printf "%20d\n", $i}' | sort -k 1 -n | head -n 1` | tee -a ../msmf-min-single
      echo -n " " | tee -a ../msmf-min-single
   done
   echo "" | tee -a ../msmf-min-single

   echo -n "${mat::-1} " | tee -a ../msmf-min-double
   echo -n `cat props | awk '{printf "%14d", $3}'` | tee -a ../msmf-min-double 
   echo -n " " | tee -a ../msmf-min-double
   for i in 5 6 7 9 11
   do
      echo -n `cat msmf | awk -v i=$i '{printf("%20d\n", $i)}' | sort -k 1 -n | head -n 1` | tee -a ../msmf-min-double
      echo -n " " | tee -a ../msmf-min-double
   done
   echo "" | tee -a ../msmf-min-double

   echo -n "${mat::-1} " | tee -a ../msmf-min-square-single
   echo -n `cat props | awk '{printf "%14d", $3}'` | tee -a ../msmf-min-square-single 
   echo -n " " | tee -a ../msmf-min-square-single
   for i in 5 6 7 8 10
   do
      echo -n `cat msmf | awk '$1 == $2' | awk -v i=$i '{printf "%20d\n", $i}' | sort -k 1 -n | head -n 1` | tee -a ../msmf-min-square-single
      echo -n " " | tee -a ../msmf-min-square-single
   done
   echo "" | tee -a ../msmf-min-square-single

   echo -n "${mat::-1} " | tee -a ../msmf-min-square-double
   echo -n `cat props | awk '{printf "%14d", $3}'` | tee -a ../msmf-min-square-double 
   echo -n " " | tee -a ../msmf-min-square-double
   for i in 5 6 7 9 11
   do
      echo -n `cat msmf | awk '$1 == $2' | awk -v i=$i '{printf("%20d\n", $i)}' | sort -k 1 -n | head -n 1` | tee -a ../msmf-min-square-double
      echo -n " " | tee -a ../msmf-min-square-double
   done
   echo "" | tee -a ../msmf-min-square-double

   cd ..
done

cd ..
