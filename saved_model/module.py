# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:54:03 2022

@author: nkayf
"""
#%% Imports and Paths

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score
import numpy as np
import datetime
import json
import re
import os

LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')

#%% Classes and Function

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def remove_tags(self, data):
        '''
        This function removes all HTML tags


        Parameters
        ----------
        data : Array
            Unprocessed data with HTML tags.

        Returns
        -------
        data : Array
            Processed data without HTML tags.

        '''
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>','',text)
        return data
    
    def lower_split(self, data):
        '''
        This function converts all letters into lowercase and split into list

        Parameters
        ----------
        data : Array
            Unprocessed data in multicases letters.

        Returns
        -------
        data : List
            Processed data in lower-case, splitted.

        '''
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()
        return data
    
    def one_hot_encoder(self,data):
        '''
        This function will encode input data using one hot encoder approach
    
        Parameters
        ----------
        input_data : List,Array
            Input Data will undergo one-hot encoding.
    
        Returns
        -------
        encoded(input_data) : Array
            Input Data will undergo one-hot encoding.
    
        '''
        enc = OneHotEncoder(sparse=False)
        return enc.fit_transform(np.expand_dims(data,axis=-1))
    
    def cat_tokenizer(self, data, token_save_path, 
                            num_words=10000, 
                            oov_token='<OOV>', prt=False):
        '''
        This function setup the list of tokenizer for the dataset

        Parameters
        ----------
        data : list
            Un-tokenised data.
        token_save_path : string
            Path to save the token file.
        num_words : TYPE, int
            Maximum words processed. The default is 10000.
        oov_token : string, optional
            DESCRIPTION. The default is '<OOV>'.
        prt : TYPE, Boolean
            To print token dictionary for quick check . The default is False.

        Returns
        -------
        data : list
            Tokenised data.

        '''
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        token_json = tokenizer.to_json()
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)
        
        # observe no of words
        word_index = tokenizer.word_index
        
        if prt==True:
            print(dict(list(word_index.items())[:10]))
        
        # vectorize sequence of txt
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def cat_pad_sequence(self,data,maxlen=200, padding='post', truncating='post'):
        '''
        This function set the paddings for each items in list


        Parameters
        ----------
       data : list
            Unpadded data.
        maxlen : TYPE, optional
            DESCRIPTION. The default is 200.
        padding : TYPE, optional
            DESCRIPTION. The default is 'post'.
        truncating : TYPE, optional
            DESCRIPTION. The default is 'post'.

        Returns
        -------
         data : Array
            Padded data.

        '''
        
        data = pad_sequences(data, maxlen=maxlen, padding=padding, truncating=truncating)
        return data

class ModelCreation():
    def lstm_layer(self,num_words,nb_categories,embedding_output=64, nodes=32, dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words,embedding_output))
        model.add(LSTM(nodes,return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(nodes))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories,activation='softmax'))
        model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')
        model.summary()
        
        return model
    
class ModelTraining():
    def model_training(self,model, x_train,y_train, validation_data,epochs=100):
        log_files = os.path.join(LOG_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss',patience=3)
        
        return model.fit(x_train,y_train, epochs=epochs, validation_data=validation_data,callbacks=[tensorboard_callback,early_stopping_callback])

class ModelEvaluation():
    def allocate_eval_data(self,X_test,model,nb_categories):
        #preallocate approach (faster)
        predicted_advanced = np.empty([len(X_test),nb_categories])
        
        for index, test in enumerate(X_test):
            predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))
        return predicted_advanced

    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true,y_pred))
        print("Confusion Matrix :\n",confusion_matrix(y_true,y_pred))
        print("Accuracy Score\t: ",accuracy_score(y_true,y_pred)*100)
        print("F1-Score\t: ",f1_score(y_true,y_pred,average='micro'))

