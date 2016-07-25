#!/bin/bash

ABHSF_ROOT="${HOME}/projects/abhsf"

cyan='\033[0;36m'
reset='\033[0m'

mkdir -p results
cd results

while read line
do
   cat=$(echo -n $line | awk '{print $1}' | tr -d [:space:])
   mat=$(echo -n $line | awk '{print $2}' | tr -d [:space:])

   echo -e "${cyan}category: ${cat}, matrix: ${mat}${reset}"

   wget http://www.cise.ufl.edu/research/sparse/MM/${cat}/${mat}.tar.gz
   tar -xzf ${mat}.tar.gz
   rm -f ${mat}.tar.gz
   mv $mat ${mat}-temp
   rm -rf $mat
   mkdir $mat
   mv ${mat}-temp/${mat}.mtx $mat/
   rm -rf ${mat}-temp
   
   cd $mat
   $ABHSF_ROOT/block_statistics/block_stats ${mat}.mtx
   rm ${mat}.mtx
   cd ..

   echo ""
done < ../$1

cd ..
