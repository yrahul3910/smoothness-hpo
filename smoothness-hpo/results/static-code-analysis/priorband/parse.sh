#!/usr/bin/env sh

grep "^Result" priorband.txt > t.txt
python3 parse.py  # prints pd-pf, pd, pf, prec

rm t.txt
