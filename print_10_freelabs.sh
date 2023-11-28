#!/bin/bash

string=$(/vol/linux/bin/freelabmachine)
while [[ $string != *"gpu"* && $string != *"ray"* ]]
do
    string=$(/vol/linux/bin/freelabmachine)
    echo $string
done
echo $string
