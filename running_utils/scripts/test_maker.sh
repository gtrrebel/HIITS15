#!/bin/bash

CODE=$1
CODEPATH="/cs/fs/home/othe/Windows/Desktop/hiit/hiit_test_input"
FILENAME="${CODEPATH}/${CODE}.runfile"
touch $FILENAME
FILESTOLOOP="${CODEPATH}/${CODE}/in/*"
for f in $FILESTOLOOP
	do
		echo ${f} >> $FILENAME
done
python "$CODEPATH/../HIITS15/running_utils/home_ukko_runner1.py" $CODE $FILENAME
rm $FILENAME
