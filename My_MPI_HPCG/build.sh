#!/bin/bash

source env.sh

ln -sf setup/Make.MY_MPI make.inc
make -j4