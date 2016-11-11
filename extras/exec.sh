#!/bin/sh
now=$(date +"%d%h%Y_%H:%M:%S")
logfile=$now
echo 'Execute:' >> $logfile
initialsec=$(date +"%s") #seconds since 1970-01-01 00:00:00 UTC
echo $now >> $logfile
#echo $initialsec
echo "Run $1:" >> $logfile
#paramstr=$(cat $1)
#echo $paramstr
execoutstr="$(python $1)"
echo $execoutstr >> $logfile
finalsec=$(date +"%s")
#echo $finalsec
exectime=$(expr $finalsec - $initialsec)
echo "execution time(seconds) = $exectime" >> $logfile
