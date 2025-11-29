#!/bin/bash
export PYTHONPATH=.;poetry run python training/execution.py $1 $2 $3
exit $?
