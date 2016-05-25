#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

cd results

for mat in */
do
   cd $mat
   cat msmf-min-single | awk -v mat=$mat '{printf "%20s %10s %20d\n", mat, $1, $4}' | sort -k 3 -n | head -n 1
   cd ..
done

cd ..
