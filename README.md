# CellDeath

### Prerequisites and Major Dependencies
* CellProfiler 2.3.1
* Python 3.7
* Tensorflow 1.14
* Anaconda

---

### Project Setup
Clone the repository
```
$ git clone https://github.com/BooneAndrewsLab/CellDeath.git
$ cd CellDeath
```

Download [input test dataset][celldeath_test_images] and save it to 'test' folder
```
$ mkdir test
$ wget -P models https://thecellvision.org/celldeath/CellDeathProject_Demo_Images.tar.gz
$ tar -xvzf test/CellDeathProject_Demo_Images.tar.gz --directory=test
```

Create two conda environments for CellProfiler and Outlier Detection steps
```
$ conda env create -f environment_cellprofiler.yml
$ conda env create -f environment_outlier_detection.yml
```

---

### Step 1. Cell Segmentation and Feature Extraction using CellProfiler
Activate conda environment
```
$ conda activate cellprofiler_2.3.1_env
```
Create output folder for CellProfiler results
```
Usage:
$ mkdir <CP_OUTPUT_FOLDER>

Example:
$ mkdir -p test/cellprofiler_results/ATG_R1
```
Run commands below for each unique screen-replicate image folder
```
Usage:
$ cellprofiler -c -r -p <PATH_TO_CELLPROFILER_PIPELINE> -i <INPUT_FOLDER> -o <CP_OUTPUT_FOLDER>
$ cellprofiler -c -r -p <PATH_TO_BATCH_FILE> -o <CP_OUTPUT_FOLDER>


Example:
$ cellprofiler -c -r -p 1_CellSegmentation_and_FeatureExtraction_Pipeline.cppipe -i test/CellDeathProject_Demo_Images/2015_April_24_Atg8_R1 -o test/cellprofiler_results/ATG_R1
$ cellprofiler -c -r -p test/cellprofiler_results/ATG_R1/Batch_data.h5 -o test/cellprofiler_results/ATG_R1
```
_NOTE: The first cellprofiler command will generate a batch file (Batch_data.h5) in the output directory. This will be 
used as the input to the second cellprofiler command. If the user wants to process image sets by batch, the user can 
add two parameters: -f <FIRST_IMAGE_SET> -l <LAST_IMAGE_SET> when running the command._

---

### Step 2. Outlier Detection
Activate conda environment
```
$ conda activate OD_env
```
Create output folder for Outlier Detection results
```
Usage:
$ mkdir <OD_OUTPUT_FOLDER>

Example:
$ mkdir -p test/OD_results/ATG_R1
```
Combine CellProfiler results
```
Usage:
$ python <PATH_TO_COMBINE_CP_SCRIPT> -p <CP_OUTPUT_FOLDER> -o <OD_OUTPUT_FOLDER> -m <GENE_MAPPING_SHEET>

Example:
$ python 2_Combine_CellProfiler_Results.py -p test/cellprofiler_results/ATG_R1 -o test/OD_results/ATG_R1 -m EMA_MappingSheet.csv
```
Run commands below for each screen-replicate-timepoint rawdata
```
Usage:
$ python 3_OutlierDetection_TerminalPhenotypes.py -i <INPUT_FILE_T##_rawdata.csv> -o <OD_OUTPUT_FOLDER> -f <FEATURE_SET>
$ python 4_OutlierDetection_WT_penetrance.py -i <INPUT_FILE_T##_rawdata.csv> -o <OD_OUTPUT_FOLDER> -f <FEATURE_SET>

Example:
$ python 3_OutlierDetection_TerminalPhenotypes.py -i test/OD_results/ATG_R1/2015_April_24_Atg8_R1_T00_rawdata.csv -o test/OD_results/ATG_R1 -f FeatureSets_TerminalPhenotypes.txt
$ python 4_OutlierDetection_WT_penetrance.py -i test/OD_results/ATG_R1/2015_April_24_Atg8_R1_T00_rawdata.csv -o test/OD_results/ATG_R1 -f FeatureSets_TerminalPhenotypes.txt
```
_NOTE: The outlier detection scripts have other parameters with set default values such as: 'threshold', 'combine-wt', 
'variance', 'distance-method', 'OneClassSVM-kernel', 'OneClassSVM-gamma', and 'probability'. Users can use these 
parameters and ran with their own custom values as needed._

### License
This software is licensed under the [BSD 3-Clause License][BSD3]. Please see the 
``LICENSE`` file for more details.

[celldeath_test_images]: https://thecellvision.org/celldeath/CellDeathProject_Demo_Images.tar.gz
[BSD3]: https://opensource.org/license/bsd-3-clause