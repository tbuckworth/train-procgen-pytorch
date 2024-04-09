#!/bin/bash
cat $1 | ssh -tt -o StrictHostKeyChecking=no $(./free_gpu)

