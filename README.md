# Preparing machine learning code for deployment!
'This has nothing to do with the real life.'


This repository demonstrates how to prepare a working python code gethered in one file into deployment ready repository. Original file 'start.py' is located in 'not_in_use_anymore' folder. 

## Steps

### 1. Transforming iris dataset into SQL database. 
The file is called 'iris_to_db.py' and is located in folder 'pre_deployment'. 

### 2. Spliting the original file into 'model_training.py' and 'predict.py'. 

### 3. Creating file 'functions.py' for additional functionality. 

### 4. Create Requirements.txt and specify packages

### 5. Write clear README file

### 6. Add Licence

## File explanation:

'predict.py' predicts iris type based on the trained model.

'functions.py' includes additional functions used to analyze repo

Predeployment folder has two scripts: 
    - 'iris_to_db.py' creates database from CSV file and save it in 'database' directory
    - 'model_training.py' trains a model and save it in 'models' directory

Folder 'not_in_use_anymore' includes starting file ('start.py') and iris csv file 
 
