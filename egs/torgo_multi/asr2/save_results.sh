#!/bin/bash

spk=new

pushd exp

pushd train_array*

    pushd results
    rm *.ep.* *.best
    popd

    pushd decode*
    rm *.json
    popd

popd

mkdir -p dog
mv train_array_head_pytorch_train_specaug dog/${spk}_multi

popd