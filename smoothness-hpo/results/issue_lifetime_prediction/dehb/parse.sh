#!/usr/bin/env bash

grep "^Result" $1.txt > t.txt
python3 parse.py
