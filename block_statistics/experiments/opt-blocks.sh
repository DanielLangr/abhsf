#!/bin/bash

cd results

rm -rf opt-bs-double
rm -rf opt-bs-single

for mat in */
do
   echo -e "matrix: ${mat}"

   cd $mat

   echo -n "${mat::-1} " | tee -a ../opt-bs-double
   echo -n "${mat::-1} " | tee -a ../opt-bs-single

   cat msmf-min-double | sort -n -k 4 | head -n 1 | awk '{print $2" "$3}' | tee -a ../opt-bs-double
   cat msmf-min-single | sort -n -k 4 | head -n 1 | awk '{print $2" "$3}' | tee -a ../opt-bs-single

   cd ..
done

cd ..
