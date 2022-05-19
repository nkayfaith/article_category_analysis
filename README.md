# Article Category Analysis
This project is a part of the assessment for SHRDC Data Science course

#### -- Project Status: [Completed]

## Project Intro/Objective
The purpose of this project is to categorise articles(English)

### Methods Used
* Inferential Statistics
* Deep Learning
* Data Visualization
* Predictive Modeling


### Technologies
* Python
* Pandas, Numpy, Sklearn
* Tensorflow, Tensorboard
* Google Colaboratory

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is retrieved from [https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv].
3. Data processing/transformation scripts are being kept [https://github.com/nkayfaith/article_category_analysis/tree/main/saved_model]

## Discussion, Analysis and Result
1. Model Architecture as follows:

A Sequential model with attributes of Dense = 256, Dropout = .3, Hidden Layer = 3, Epochs = 100 with callbacks
![image](statics/model.png)


2. Training processes recorded as follows:

Process :
![image](statics/train_process.png)

Loss :
![image](statics/loss.png)


Accuracy :
![image](statics/accuracy.png)

3. Performance of the model and the reports as follows:

Both F1 and Accuracy recorded at 
![image](statics/performance.png)

4. Reviews

## Credits
https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv
