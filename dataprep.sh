#!/bin/bash
currdir=`dirname $0`
cd $currdir

cd data
unzip images_training_rev1.zip

cd images_training_rev1
mkdir -p ../images_training_rev1_12/

find "." -name "*.jpg" | xargs -I {} echo "convert {} -resize 12x12 ../images_training_rev1_12/{}" >commands.txt
cat commands.txt|parallel
rm -f commands.txt
