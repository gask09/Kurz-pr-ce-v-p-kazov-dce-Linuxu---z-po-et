#!/bin/bash

# Set data directories
WORKDIR="tagger_competition" 
DATADIR="/storage/brno2/home/${LOGNAME}" # Or other storage

# Clean-up of SCRATCH (it is temporal directory created by server) - the commands will be launched on the end when the job is done
trap 'clean_scratch' TERM EXIT
trap 'cp -a "${SCRATCHDIR}" "${DATADIR}"/ && clean_scratch' TERM

# Prepare the task - copy all needed files from working directory into particular computer which will finally do the calculations
cp -a "${DATADIR}"/"${WORKDIR}"/* "${SCRATCHDIR}"/ || exit 1 # If it fails, exit script

# Change working directory - script goes to the directory where calculations are done
cd "${SCRATCHDIR}"/ || exit 1 # If it fails, exit script


# if python3 venv (dir containing python3 virtual enviroment) doesn't exist, create it
module add python/3.10.4-intel-19.0.4-sc7snnf
if [! -d venv]; then
	python3 -m venv venv
fi 

# TMPDIR is way too small (~1 GB) to install CUDA-version of Keras/PyTorch
export TMPDIR=$SCRATCHDIR

# Upgrading piptools and installing all the packages. If they are installed, pip should just skip it, confirming that they are here
venv/bin/pip install --no-cache-dir --upgrade pip setuptools
venv/bin/pip install --no-cache-dir keras~=3.0.5 --extra-index-url=https://download.pytorch.org/whl/cu118 torch~=2.2.0 torchaudio~=2.2.0 torchvision~=0.17.0 torchmetrics~=1.3.1 flashlight-text~=0.0.3 tensorboard~=2.16.2

# Run the task with installed python3 venv
venv/bin/python3 ,"$SCRATCHDIR"/tagger_competition.py

# Copy results back to home directory
# That might be too slow if model weights (900 MB+) are saved too, so it might be better idea to use scp
cp -a "${SCRATCHDIR}" "${DATADIR}"/"${WORKDIR}" || export CLEAN_SCRATCH=false

exit
