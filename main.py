# ==========================================
# Titanic Survival Prediction Project
# ==========================================

# ==========================================
# 👤 Member 1 :Sanket: Import Libraries & Load Data
# Task: Load dataset and check basic info
# ==========================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("First 5 rows:\n", train_df.head())
print("\nMissing Values:\n", train_df.isnull().sum())


# ==========================================
# 👤 Member 2:Prajwal: Data Preprocessing
# Task: Clean data and handle missing values
# ==========================================
# Save PassengerId
test_ids = test_df['PassengerId']

# Drop unnecessary columns
train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

# Fill missing values
for df in [train_df, test_df]:
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())




# ==========================================
# 👤 Member 3: Ranjit : Feature Engineering
# Task: Create new useful features
# ==========================================



for df in [train_df, test_df]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0




# ==========================================
# 👤 Member 4:Smarak : Encoding
# Task: Convert categorical data into numeric
# ==========================================







# ==========================================
# 👤 Member 5:Snehith Data Splitting & Scaling
# Task: Split data and normalize it
# ==========================================







# ==========================================
# 👤 Member 6: Anurag: Model Training & Cross Validation
# Task: Train model and validate performance
# ==========================================









# ==========================================
# 👤 Member 7: Dev : Prediction & Evaluation
# Task: Evaluate model performance
# ==========================================
