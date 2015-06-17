#!/bin/bash

CODE=$1
CODEPATH="~/Windows/Desktop/hiit/hiit_test_input"
touch "$CODEPATH/$CODE.run.tmp"
for f in "$CODEPATH/$CODE/in/*"
	do
		$f"\n" >> "$CODEPATH/$CODE.run.tmp"
done
