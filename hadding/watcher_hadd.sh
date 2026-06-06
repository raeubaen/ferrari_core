option="beam"

HADD_NOW_DIRS="$MAIN_FOLDER/to_hadd_now.txt"
HADD_GLOB_BUFFER="$MAIN_FOLDER/to_hadd_buffer.txt"

while true; do
  if [ -e ${HADD_NOW_DIRS} ]; then
    echo $HADD_NOW_DIRS exists, starting
    timeout 100s bash hadd.sh > $PLOT_MAIN_FOLDER/logs/hadd_$(date | sed 's/ /-/g') 2>&1;
  fi
  echo $HADD_NOW_DIRS not found, sleeping 2 seconds and retrying
  echo "for DEBUG, cat of ${HADD_GLOB_BUFFER}"
  cat ${HADD_GLOB_BUFFER}
  sleep 2
done
