#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

declare -a formats=("COO-32" "CSR-32" "Bcoo" "Bcsr" "Bbitmap" "Bdense32" "Bdense64" "Babhsf32" "Babhsf64")

cd results

for mat in */
do
   echo -e "${cyan}matrix: ${mat}${reset}"

   cd $mat

   rm -rf msmf-min

   for i in {3..11}
   do
      j=$(($i-3))
      f=${formats[$j]}
      cat msmf | awk -v f=$f -v i=$i '{printf("%10s %6d %6d %20d\n", f, $1, $2, $i)}' | sort -k 4 -n | head -n 1 | tee -a msmf-min
   done

   cd ..
done

cd ..
