#!/usr/bin/env bash

grep "^Result" priorband.txt > t.txt
python3 parse.py

rm t.txt
