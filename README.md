##This directory contains codes of the experiment related to the fact-checking and claim verification on the FEVER dataset.
##The details of the project is published at ROMCIR'21
##The data folder contains sample data to test the code. For the entire dataset, follow the below-mentioned link: 
##https://fever.ai/resources.html

## Requirements
* **PyTorch r1.1.0**
* Python 3.5+
* CUDA 8.0+ (For GPU)

## File
* bert.py: Classfication using BERT 
* cnn.py: Classfication using CNN 
* lstm.py: Classfication using LSTM 
* randomforest.py: Classfication using Randomforest 
* sdg.py: Classfication using Stochastic Gradient Descent  
* svm.py: Classfication using Support Vector Machine

## Instructions to run the programs
## To run bert.py, cnn.py, lstm.py, randomforest.py, sdg.py, svm.py use the below command:
	python <filename>