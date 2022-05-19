# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:55:55 2022

@author: nkayf
"""
#%% Imports and Paths
import pandas as pd
import numpy as np
import os
from module import ExploratoryDataAnalysis, ModelCreation,ModelTraining, ModelEvaluation
from sklearn.model_selection import train_test_split


TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')

#%% Step 1) Data Loading

URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(URL)
txt_dat = df['text']
cat = df['category']

#%% Step 2) Data Interpretation/Inspection

df.info()

#%% Step 3) Data Cleaning

# Change all cases to lower
eda = ExploratoryDataAnalysis()
txt_dat = eda.lower_split(txt_dat)

# =============================================================================
# Standardise token cases
# =============================================================================

#%% Step 4) Feature Selection
#%% Step 5) Data Preprocessing

# Data Vectorization
txt_dat = eda.cat_tokenizer(txt_dat,TOKENIZER_PATH)
txt_dat = eda.cat_pad_sequence(txt_dat)

# Encode
cat = eda.one_hot_encoder(cat)
nb_categories = cat.shape[1]

# =============================================================================
# Tokenize data with max. words processed = 10000
# Add padding with maxlen = 200
# Encode label using OHE
# =============================================================================

#%% Step 6) Model Building

mc = ModelCreation()
num_words = 10000
model = mc.lstm_layer(num_words,nb_categories,embedding_output=128, nodes=64, dropout=0.2)

# =============================================================================
# embedding_output=128, nodes=64, dropout=0.2, hidden_layer=2
# =============================================================================

#%% Step 7) Model Training

X_train, X_test, y_train, y_test = train_test_split(txt_dat, cat, test_size=.3, random_state=123)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

mt = ModelTraining()
hist = mt.model_training(model, X_train,y_train, (X_test,y_test),epochs=100)
print(hist.history.keys())

# =============================================================================
# epochs = 100 with callbacks
# =============================================================================

#%% Step 8) Model Performance
#%% Step 9) Model Evaluation

me = ModelEvaluation()
predicted_advanced = me.allocate_eval_data(X_test,model,nb_categories)
y_pred = np.argmax(predicted_advanced, axis=-1)
y_true = np.argmax(y_test, axis=-1)
me.report_metrics(y_true,y_pred)

# =============================================================================
# Accuracy recorded at 86.67%
# Graph shows high accuracy, low loss which indicates model is good enough
# =============================================================================

#%% Step 10) Model Deployment
model.save(MODEL_PATH)



