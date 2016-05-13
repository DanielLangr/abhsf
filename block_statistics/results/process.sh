#!/bin/bash

while read line
do
   cat=$(echo $line | cut -f1 -d\ )
   mat=$(echo $line | cut -f2 -d\ )

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
done < $1
