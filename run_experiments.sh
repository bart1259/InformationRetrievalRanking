
# pip install nbconvert
jupyter nbconvert --to script ExperimentNotebook3.ipynb

####### KEYBOARD DISTANCE AUGMENTATION ########

export AUGMENTATION="synonym"
export DATASET_NAME="walmart-amazon"

python ExperimentNotebook3.py

export DATASET_NAME="news-aggregator"

python ExperimentNotebook3.py

export DATASET_NAME="covid-abstracts"

python ExperimentNotebook3.py

############### NO AUGMENTATION ################

export AUGMENTATION="none"
export DATASET_NAME="walmart-amazon"

python ExperimentNotebook3.py

export DATASET_NAME="news-aggregator"

python ExperimentNotebook3.py

export DATASET_NAME="covid-abstracts"

python ExperimentNotebook3.py