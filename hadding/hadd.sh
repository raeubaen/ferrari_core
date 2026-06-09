#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "DEBUG: Running .sh inside $SCRIPT_DIR"

start=$(date +%s%3N)  # milliseconds

option="beam"

HADD_NOW_DIRS="${PLOT_MAIN_FOLDER}/to_hadd_now.txt"
HADD_GLOB_BUFFER="${PLOT_MAIN_FOLDER}/to_hadd_buffer.txt"

if [ ! -f "${HADD_NOW_DIRS}" ]; then
    echo "no files to hadd in $HADD_NOW_DIRS"
    return
fi

RUN_NO=$(tail -n 1 "${HADD_NOW_DIRS}" \
    | awk -F 'run_' '{print $2}' \
        | awk -F '/' '{print $1}')

echo "RUN_NO = ${RUN_NO}"

ALL_SPILL_DIR="${PLOT_MAIN_FOLDER}/run_${RUN_NO}/${option}_all_spill"

FIRST_SPILL=$(grep "run_${RUN_NO}" "${HADD_NOW_DIRS}" | head -n 1)

echo "mkdir -p ${ALL_SPILL_DIR}"

mkdir "${ALL_SPILL_DIR}"

echo "hadding (output in /dev/null - to debug open the code...)"

# Copy helper php files if needed
cp ${PHP_FILES_DIR}/*.php "${ALL_SPILL_DIR}/"

FILES=$(grep "run_${RUN_NO}" "${HADD_NOW_DIRS}" \
    | awk '{print $1 "/histos.root"}' \
        | tr '\n' ' ')

DEST="${ALL_SPILL_DIR}/histos.root"

echo "Input files:"
echo "${FILES}"

if [ -e "${DEST}" ]; then

  echo "${DEST} exists, checking if corrupt"
  time root -l -b -q "${SCRIPT_DIR}/fileCheck.C(\"${DEST}\")" | grep "FILE_OK"

  if [ $? -ne 0 ]; then
           echo "${DEST} is corrupt or zombie, skipping"
           echo "Removing hadded file..."
           rm -f "${DEST}"
  fi
else
  echo "${DEST} does not exist, will be created"
fi

for CURRENT_FILE in ${FILES}; do

      echo current Iteration: ${CURRENT_FILE}
      time root -l -b -q "${SCRIPT_DIR}/fileCheck.C(\"${CURRENT_FILE}\")" | grep "FILE_OK"

          if [ $? -ne 0 ]; then
               echo "${CURRENT_FILE} is corrupt or zombie, skipping"
          else
               if [ -e "${DEST}" ]; then
                  echo "${DEST} exists, appending!"
                  echo "Appending to existing ${DEST}"
                  time hadd -a "${DEST}" ${CURRENT_FILE}
                else
                  echo "Creating ${DEST}"
                  echo "hadd -f "${DEST}" ${CURRENT_FILE}"
                  time hadd -f "${DEST}" ${CURRENT_FILE}
                fi
          fi
done

PLOTLIST=$(find $(cat $HADD_NOW_DIRS) -maxdepth 1 -type f -name "*.csv" 2>/dev/null | head -n 1)

cd ${WORKING_DIR}

python3 -m ferrari_core.hadding.plot_hadded -po $ALL_SPILL_DIR/ -pl $PLOTLIST

# Update bookkeeping files
if diff "${HADD_NOW_DIRS}" "${HADD_GLOB_BUFFER}" > /dev/null 2>&1; then

    rm -f "${HADD_GLOB_BUFFER}"
    rm -f "${HADD_NOW_DIRS}"

else

    grep -Fvx -f "${HADD_NOW_DIRS}" "${HADD_GLOB_BUFFER}" \
            | grep "run_${RUN_NO}" > /tmp/hadd_temp

    rm -f "${HADD_NOW_DIRS}"
        mv /tmp/hadd_temp "${HADD_GLOB_BUFFER}"

fi

end=$(date +%s%3N)
elapsed=$((end - start))

echo "Elapsed time: ${elapsed} ms"
echo "----------------- hadd done -----------------"
