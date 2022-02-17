#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -d Dataset"
   echo -e "\t-d The dataset on which the model should be trained, must be in [arrhythmia, thyroid, kdd, kddrev]"
   exit 1 # Exit script after printing help
}

while getopts "m:d:" opt
do
   case "$opt" in
      d ) Dataset="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$Dataset" ]; then
   echo "Dataset should be specififed";
   helpFunction
fi

if ! [[ "$Dataset" =~ ^(arrhythmia|thyroid|kdd|kddrev)$ ]]; then
   echo "Chosen dataset is not available";
   helpFunction
fi

# Begin script in case all parameters are correct
echo "TracinAD on dataset $Dataset"

if [[ $Dataset == "arrhythmia" ]]; then
    bash ./VAE/batch/arrhythmia.sh
fi
if [[ $Dataset == "thyroid" ]]; then
    bash ./VAE/batch/thyroid.sh
fi
if [[ $Dataset == "kdd" ]]; then
    bash ./VAE/batch/kdd.sh
fi
if [[ $Dataset == "kddrev" ]]; then
    bash ./VAE/batch/kddrev.sh
fi
