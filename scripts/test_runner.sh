#!/bin/bash

INFILE=$1
RUNFILE=`basename $1`
MYHOST=`hostname`
MYDATE=`date +%Y-%m-%d.%H:%M:%S`
LOG="$HOME/Windows/Desktop/hiit/hiit_test_results/$RUNFILE.$MYHOST.$MYDATE.$$"
python $INFILE >> $LOG 2> /dev/null

