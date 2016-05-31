#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

rm -rf msmf-bs-adaptive-double
rm -rf msmf-bs-adaptive-single
rm -rf msmf-bs-minfixed-double
rm -rf msmf-bs-minfixed-single

rm -rf msmf-bs-minfixed-134-double
rm -rf msmf-bs-minfixed-13-double

for mat in */
do
 # echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   echo -n "${mat::-1} " | tee -a ../msmf-bs-adaptive-double
   echo -n "${mat::-1} " | tee -a ../msmf-bs-minfixed-double
   echo -n "${mat::-1} " | tee -a ../msmf-bs-minfixed-134-double
   echo -n "${mat::-1} " | tee -a ../msmf-bs-minfixed-13-double

   for i in $(seq 1 64)
   do
      echo -n `sed "${i}q;d" msmf | awk '{print $11}'` | tee -a ../msmf-bs-adaptive-double
      echo -n " " | tee -a ../msmf-bs-adaptive-double

      echo -n `sed "${i}q;d" msmf | awk '{m=$9;for(i=5;i<8;i++)if($i<m)m=$i;print m}'` | tee -a ../msmf-bs-minfixed-double
      echo -n " " | tee -a ../msmf-bs-minfixed-double

      echo -n `sed "${i}q;d" msmf | awk '{m=$5;if($7<m)m=$7;if($9<m)m=$9;print m}'` | tee -a ../msmf-bs-minfixed-134-double
      echo -n " " | tee -a ../msmf-bs-minfixed-134-double

      echo -n `sed "${i}q;d" msmf | awk '{m=$5;if($7<m)m=$7;print m}'` | tee -a ../msmf-bs-minfixed-13-double
      echo -n " " | tee -a ../msmf-bs-minfixed-13-double
   done
   echo "" | tee -a ../msmf-bs-adaptive-double
   echo "" | tee -a ../msmf-bs-minfixed-double
   echo "" | tee -a ../msmf-bs-minfixed-134-double
   echo "" | tee -a ../msmf-bs-minfixed-13-double

   cd ..
done

cd ..
