HADD_NOW_DIRS="$PLOT_MAIN_FOLDER/to_hadd_now.txt"
HADD_GLOB_BUFFER="$PLOT_MAIN_FOLDER/to_hadd_buffer.txt"

while true; do
  if [ -e ${HADD_NOW_DIRS} ]; then
    echo $HADD_NOW_DIRS exists, starting
    timeout 40s bash hadd.sh 2>&1 | tee $PLOT_MAIN_FOLDER/logs/hadd_$(date | sed 's/ /-/g');
  fi
  echo $HADD_NOW_DIRS not found, sleeping 2 seconds and retrying
  echo "for DEBUG, cat of ${HADD_GLOB_BUFFER}"
  cat ${HADD_GLOB_BUFFER}
  sleep 2
done
