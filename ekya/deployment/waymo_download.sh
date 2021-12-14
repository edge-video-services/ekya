#!/bin/bash
OUTPUTDIR=/home/researcher/datasets
FILENAMES="1uT89nje_nRziWhAEdwgbigKRX2nP279l,waymo0.tar
18Oj3CEoLAGZddZRCzOQLSqhhuRqHyENN,waymo1.tar
1nM5HIaXQ3r1cQP7LsVBbRKo6QHECWEel,waymo2.tar
1qJdOUHwKsdKSaO-RZadAXTlNJdtZmGpU,waymo3.tar
1wPL2isxytJx-IOc-Wfz2zthVbGPUbFBU,waymo4.tar
1Ozv8W2OPF0R9Y8VDDx99RSDgKM8Ssglw,waymo5.tar
1XwCwPKLN_9GvHMRC2rw6HfAasG0GUWpM,waymo6.tar
1FXgYsou3tpio3y3L_6wkF-IZqs3ZlxEJ,waymo7.tar"


for i in $FILENAMES; do
  IFS=","
  set -- $i
  FILEID=$1
  FILENAME=$2
  echo $FILEID
  echo $FILENAME
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${OUTPUTDIR}${FILENAME} && rm -rf /tmp/cookies.txt
done

for i in {1..7}; do
  tar -xvzf ${OUTPUTDIR}waymo${i}.tar --strip-components=3 -C ${OUTPUTDIR}
done