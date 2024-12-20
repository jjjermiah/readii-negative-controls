#!/bin/bash

# Get series metadata
# python workflow/scripts/find-series.py

NUM_SERIES=all

# Download series
# DATASET_LIST=(HEAD-NECK-RADIOMICS-HN1)
DATASET_LIST=(RADCURE)
# DATASET_LIST=(HNSCC)
# DATASET_LIST=(RADCURE HNSCC HEAD-NECK-RADIOMICS-HN1)

download_series() {
  local dataset=$1
  local num_series=$2
  python workflow/scripts/download-series.py -f metadata/$dataset.json --num-series $num_series
}
export -f download_series


unzip_dirs() {
  local dataset=$1
  local num_series=$2

  # download_series $dataset $num_series

  local tmp_dir=$3/$dataset/images/dicoms
  local real_dir=$4/$dataset/images/dicoms
  mkdir -p $tmp_dir
  mkdir -p $real_dir

  python workflow/scripts/unzip_dirs.py --overwrite rawdata/zipped/$dataset ${tmp_dir}

  if [ -d $real_dir ]; then
    rm -rf $real_dir
    mkdir -p $real_dir
  fi


	subdirectories=$(find $tmp_dir -mindepth 1 -maxdepth 1 -type d)

	echo "Found $(echo "$subdirectories" | wc -l) subdirectories in $tmp_dir"
	echo "$subdirectories" | parallel -j 30 --bar \
		'imgtools -v dicomsort \
			--action move \
			{} '"${real_dir}"'/%PatientID/StudyUID-%StudyInstanceUID/%Modality_SeriesUID-%SeriesInstanceUID/'

	# imgtools -v dicomsort \
  #   --action move \
  #   -j 30 \
  #   $tmp_dir \
  #   ${real_dir}/%PatientID/StudyUID-%StudyInstanceUID/%Modality_SeriesUID-%SeriesInstanceUID/
}
export -f unzip_dirs


tmp_dir=rawdata/tmp
real_dir=rawdata
mkdir -p $tmp_dir
mkdir -p $real_dir

# parallel download_series ::: "${DATASET_LIST[@]}"
# parallel unzip_dirs ::: "${DATASET_LIST[@]}" ::: $NUM_SERIES ::: $tmp_dir ::: $real_dir 

for dataset in "${DATASET_LIST[@]}"; do
  unzip_dirs $dataset $NUM_SERIES $tmp_dir $real_dir
done

# for dataset in "${DATASET_LIST[@]}"; do
#   download_series $dataset
#   unzip_dirs $dataset $tmp_dir $real_dir
# done

# rm -rf $tmp_dir