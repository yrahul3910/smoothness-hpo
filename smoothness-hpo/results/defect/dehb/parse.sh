#!/usr/bin/env sh

grep "^\[\d" dehb.txt > t.txt
python3 parse.py  # prints f1, pd, pf, prec
