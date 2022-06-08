# Crowd for Good - Cancer ML Benchmarking Challenge
### submitter : [pjmathematician](https://www.topcoder.com/members/pjmathematician)

## Deployment

### Install required packages 
`pip install -r requirements.txt`

### Running the program
Set the project directory as the working directory and execute
```shell
> python main.py
```

The submission csv file is saved in ./solution/submission.csv and the model trained (for reference) is saved in ./models

## Solution Approach

### Introduction
Tried and tested many models and found XGBoost the fastest and the most accurate and tuned the hyperparameters to maximize the accuracy.
All the models were yielding 98% + accuracy with some tuning, so I went ahead with XGBoost as it was the fastest.



### EDA and train test split
![](https://raw.githubusercontent.com/pj-mathematician/Crowd-for-Good---Cancer-ML-Benchmarking-Challenge/ed6863c08d3d4a139f5684fb7c3daac977e8d524/images/image.png)

The training dataset had an unequal distribution of the 1 and 0 labels so I used Stratified KFold to generate train test sets with similar distribution of labels from the training data, and used them throughout this project to evaluate different models.

Plotted a correlation plot and found that the features are present in clusters with respect to each other. 
![](https://github.com/pj-mathematician/Crowd-for-Good---Cancer-ML-Benchmarking-Challenge/blob/main/images/image.png?raw=true)
These clusters suggested that RandomForestClassification or ExtraTreesClassification will work the best for this. 

### Modelling 
With initial models being RandomForest and ExtraTrees, they produced a very accurate result in the very first iteration yielding 90%+ accuracy. From this I moved to gradient boosting models with RandomForest Ensemble (Default for XGBoost). 

Modelling with XGBoostClassifier generated even more accurate (97%+) results and I decided to move ahead with it.

I tried many many different models including deep learning models implemented in PyTorch (TabNet) and purely sequential CNNs implemented in Keras. They generated great results but the training time was very high compared to the machine learning models (3-4 minutes with deep learning models and 3-5 seconds for machine learning models)

After training various models, I plotted feature importance plots and decided to filter the features accordingly. 
![](https://i.imgur.com/24hpamN.png)
But training the data based on the K best features was not contributing much towards increasing the accuracy. And hence I decided not to move ahead with feature selection and used all the features for the final training instead.

### Hyperparameter Tuning for XGBoost
Increasing the number of estimators and leaves was causing overfitting of the data, which is also supported by the volume of training data that was given (360 rows). So I kept these parameters on default and tuned the learning rate a bit low as I noticed that it was increasing the accuracy.

I used the evaluation metric while fitting the data as AUC (as the final evaluation technique will also be ROC AUC).

### Final Modelling
I trained my final model on the whole training dataset with the same configurations and tuned the parameters a bit more with respect to the extremity of the prediction probabilities (which was going to be evaluated). When any of the rows was producing a probability in range 0.2 - 0.8 it showed that the model was not "confident" enough to predict that row. I accordingly changed the params to provide a more "confident" (probabilities ranging under 0.1 or more than 0.9) prediction.




 




