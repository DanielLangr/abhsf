#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

rm -rf mmf-bs-adaptive-double
rm -rf mmf-bs-adaptive-single
rm -rf mmf-bs-adaptive-134-double
rm -rf mmf-bs-adaptive-134-single
rm -rf mmf-bs-minfixed-double
rm -rf mmf-bs-minfixed-single
rm -rf mmf-bs-minfixed-134-double
rm -rf mmf-bs-minfixed-134-single

for mat in */
do
 # echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   echo -n "${mat::-1} " | tee -a ../mmf-bs-adaptive-double
   echo -n "${mat::-1} " | tee -a ../mmf-bs-adaptive-single
   echo -n "${mat::-1} " | tee -a ../mmf-bs-adaptive-134-double
   echo -n "${mat::-1} " | tee -a ../mmf-bs-adaptive-134-single
   echo -n "${mat::-1} " | tee -a ../mmf-bs-minfixed-double
   echo -n "${mat::-1} " | tee -a ../mmf-bs-minfixed-single
   echo -n "${mat::-1} " | tee -a ../mmf-bs-minfixed-134-double
   echo -n "${mat::-1} " | tee -a ../mmf-bs-minfixed-134-single

   for i in $(seq 1 64)
   do
      echo -n `sed "${i}q;d" mmf-double | awk '{print $9}'` | tee -a ../mmf-bs-adaptive-double
      echo -n " " | tee -a ../mmf-bs-adaptive-double

      echo -n `sed "${i}q;d" mmf-single | awk '{print $9}'` | tee -a ../mmf-bs-adaptive-single
      echo -n " " | tee -a ../mmf-bs-adaptive-single

      echo -n `sed "${i}q;d" mmf-double | awk '{print $10}'` | tee -a ../mmf-bs-adaptive-134-double
      echo -n " " | tee -a ../mmf-bs-adaptive-134-double

      echo -n `sed "${i}q;d" mmf-single | awk '{print $10}'` | tee -a ../mmf-bs-adaptive-134-single
      echo -n " " | tee -a ../mmf-bs-adaptive-134-single

      echo -n `sed "${i}q;d" mmf-double | awk '{m=$5;for(i=6;i<=8;i++)if($i<m)m=$i;print m}'` | tee -a ../mmf-bs-minfixed-double
      echo -n " " | tee -a ../mmf-bs-minfixed-double

      echo -n `sed "${i}q;d" mmf-single | awk '{m=$5;for(i=6;i<=8;i++)if($i<m)m=$i;print m}'` | tee -a ../mmf-bs-minfixed-single
      echo -n " " | tee -a ../mmf-bs-minfixed-single

      echo -n `sed "${i}q;d" mmf-double | awk '{m=$5;for(i=7;i<=8;i++)if($i<m)m=$i;print m}'` | tee -a ../mmf-bs-minfixed-134-double
      echo -n " " | tee -a ../mmf-bs-minfixed-134-double

      echo -n `sed "${i}q;d" mmf-single | awk '{m=$5;for(i=7;i<=8;i++)if($i<m)m=$i;print m}'` | tee -a ../mmf-bs-minfixed-134-single
      echo -n " " | tee -a ../mmf-bs-minfixed-134-single
   done

   echo "" | tee -a ../mmf-bs-adaptive-double
   echo "" | tee -a ../mmf-bs-adaptive-single
   echo "" | tee -a ../mmf-bs-adaptive-134-double
   echo "" | tee -a ../mmf-bs-adaptive-134-single
   echo "" | tee -a ../mmf-bs-minfixed-double
   echo "" | tee -a ../mmf-bs-minfixed-single
   echo "" | tee -a ../mmf-bs-minfixed-134-double
   echo "" | tee -a ../mmf-bs-minfixed-134-single

   cd ..
done

cd ..
