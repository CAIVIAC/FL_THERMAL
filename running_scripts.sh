#!/bin/bash
# pip install flaml
# pip install "flaml[automl]"
# pip install openml

# tensorboard
# pip install tensorboard --upgrade
# tensorboard --logdir=/home/djchen/Projects/FederatedLearning/FL_THERMAL/logs  --port 1234 --load_fast false

# data preprocessing
python ./methods/preparedata.py

# methods
python ./methods/automl_rForest.py
python ./methods/automl_xgboost.py
python ./methods/seq2seq_v1.py
python ./methods/seq2seq_v2.py
