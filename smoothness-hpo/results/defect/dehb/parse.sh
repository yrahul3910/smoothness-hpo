#!/usr/bin/env sh

grep "^Result" dehb.txt > t.txt
python3 parse.py  # prints pd-pf, f1
