#!/bin/bash

N=12


for dir in $1/*; do
    (
        BASEDIR=$(basename $dir)
        OUTPUT_FILE=$2/$BASEDIR.csv
        if [ -f $OUTPUT_FILE ]; then
            echo "$OUTPUT_FILE already exists."
        fi


        # TODO Convert to list statement
        if [ "$BASEDIR" == "dbase" ]; then
            continue
        fi

        if [ "$BASEDIR" == "incoming" ]; then
            continue
        fi

        if [ "$BASEDIR" == "printer_files" ]; then
            continue
        fi

        echo "Working on $BASEDIR"

        dicomtocsv $dir -k PatientID -k StudyDate -k StudyTime -k StudyDescription -k StudyInstanceUID -k AccessionNumber -k Modality -k SeriesNumber -k SeriesDescription -k SeriesInstanceUID -k Rows -k Columns -k NumberOfReferences -k InstanceNumber -k SOPClassUID -k SOPInstanceUID -k ReferencedFileID --image -o $OUTPUT_FILE
    ) &

    # allow only to execute $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -gt $N ]]; then
        # wait only for first job
        wait -n
    fi
done

# wait for pending jobs
wait
echo "Completed."