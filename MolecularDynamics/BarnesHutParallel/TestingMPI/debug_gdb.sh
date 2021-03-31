#!/usr/bin/env bash

mpirun -np 3 xterm -e gdb bin/runner -x initPipe.gdb
