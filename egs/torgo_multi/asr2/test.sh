#!/bin/bash
# decode_cycle="test"
# for rtask in ${decode_cycle}; do
#     echo ${rtask:9:1000}
# done

for x in data_org/F01/train data_org/F01/valid data_org/F01/test; do
   echo data/${x}
done