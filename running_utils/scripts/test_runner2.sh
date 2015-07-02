#!/bin/bash

INFILE=$1
INPUTFILE="$2 $3 $4 $5 $6"
RUNFILE=`basename $1`
MYHOST=`hostname`
MYDATE=`date +%Y-%m-%d.%H:%M:%S`

if [ ! -d "$HOME/Windows/Desktop/hiit/hiit_test_results/$RUNFILE" ]; then
	mkdir $HOME/Windows/Desktop/hiit/hiit_test_results/$RUNFILE
fi

LOG="$HOME/Windows/Desktop/hiit/hiit_test_results/$RUNFILE/$RUNFILE.$MYHOST.$MYDATE.$$"
python $INFILE $INPUTFILE >> $LOG 2> /dev/null

