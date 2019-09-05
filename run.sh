#!/bin/bash
hash ffplay 2>/dev/null || { echo >&2 "ffplay is not installed Aborting."; exit 1; }
if ! cmp --silent venv/requirements.txt requirements.txt; then
    echo "Creating virtual environment"
    rm -r venv/
    python3 -m venv venv/
    venv/bin/pip install -r requirements.txt
    cp requirements.txt venv/
fi
venv/bin/python $@ | ./stream.sh
