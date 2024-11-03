#!/usr/bin/env bash

grep "^Result" log.txt > t.txt
python3 parse.py
rm t.txt
