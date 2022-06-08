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

Tried and tested many models and found XGBoost the fastest and the most accurate and tuned the hyperparameters to maximize the accuracy.
All the models were yielding 98% + accuracy with some tuning, so I went ahead with XGBoost as it was the fastest.

The implementation of other models can be found in this [notebook](https://colab.research.google.com/github/pj-mathematician/Crowd-for-Good---Cancer-ML-Benchmarking-Challenge/blob/main/c4g_1_1.ipynb) (Disclaimer: RAW without documentation.)

