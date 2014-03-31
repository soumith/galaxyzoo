#!/bin/bash
currdir=`dirname $0`
cd $currdir

cd data
unzip images_training_rev1.zip

cd images_training_rev1
mkdir -p ../images_training_rev1_128/

find "." -name "*.jpg" | xargs -I {} echo "convert {} -resize 128x128 ../images_training_rev1_128/{}" >commands.txt
cat commands.txt|parallel
rm -f commands.txt

cd ../

unzip images_test_rev1.zip
cd images_test_rev1
mkdir -p ../images_test_rev1_128/

find "." -name "*.jpg" | xargs -I {} echo "convert {} -resize 128x128 ../images_test_rev1_128/{}" >commands.txt
cat commands.txt|parallel
rm -f commands.txt
