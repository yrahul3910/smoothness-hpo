#!/usr/bin/env bash

grep "^Result" priorband_2class.txt > t.txt
python3 parse.py

rm t.txt
