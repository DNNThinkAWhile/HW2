#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    exit -1
fi

train=$1
train_n=$2
test=$3
test_n=$4
stat=statistics.txt
statnpy=stat.npy

python normalize.py $train $stat $train_n
python save_stat.py $stat $statnpy
python normalize_test.py $test $statnpy $test_n
