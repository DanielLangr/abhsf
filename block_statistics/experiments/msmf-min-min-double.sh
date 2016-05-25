#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

for mat in */
do
   cd $mat
   cat msmf-min-double | awk -v mat=$mat '{printf "%20s %10s %5d x %5d %20d\n", mat, $1, $2, $3, $4}' | sort -k 6 -n | head -n 1
   cd ..
done

cd ..
