# Imports
import warnings

warnings.filterwarnings("ignore")

import xgboost as xgb
import pandas as pd

# Loading datasets
train_features = pd.read_csv('data/features_train.csv', header=None)
train_labels = pd.read_csv('data/labels_train.csv', header=None)
test_features = pd.read_csv('data/features_test.csv', header=None)

columns = [str(i) for i in range(train_features.shape[1])]
test_features.columns = columns
train_features.columns = columns
train_labels.columns = ['target']

# Using XGBoost classifier for modelling with hyperparameter tuning
model = xgb.XGBClassifier(learning_rate=0.08)
model.fit(train_features, train_labels, verbose=20, eval_metric='auc')

# Saving the model
model.save_model('./models/model.model')
print("Model saved at location models/model.model")

# Predicting the probabilities of the test features and saving it
submission = pd.DataFrame(model.predict(test_features))
submission[1] = model.predict_proba(test_features)[:, -1]
submission.to_csv('./solution/submission.csv', index=False, header=False)
print("Submission saved at location solution/submission.csv")
