#!/bin/bash

cyan='\033[0;36m'
reset='\033[0m'

while read line
do
 # cat=$(echo $line | cut -f1 -d\ )
 # mat=$(echo $line | cut -f2 -d\ )
   cat=$(echo -n $line | awk '{print $1}' | tr -d [:space:])
   mat=$(echo -n $line | awk '{print $2}' | tr -d [:space:])

   echo -e "${cyan}category: ${cat}, matrix: ${mat}${reset}"

   wget http://www.cise.ufl.edu/research/sparse/MM/${cat}/${mat}.tar.gz
   tar -xzf ${mat}.tar.gz
   rm -f ${mat}.tar.gz
   mv $mat ${mat}-temp
   mkdir $mat
   mv ${mat}-temp/${mat}.mtx $mat/
   rm -rf ${mat}-temp
   
   cd $mat
   ../../block_stats ${mat}.mtx
   rm ${mat}.mtx
   cd ..

   cd $mat
   ../../msmf
   cd ..
done < $1
