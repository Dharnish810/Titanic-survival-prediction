import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for displaying plots

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report)

# Load the Titanic dataset
data_path = r'C:\Users\Dharnish\Downloads\Titanic-Dataset.csv'
titanic_data = pd.read_csv(data_path)
print(titanic_data.head())

# Feature engineering
def create_features(df):
    df['FamilyCount'] = df['SibSp'] + df['Parch'] + 1
    df['IsSingle'] = (df['FamilyCount'] == 1).astype(int)
    df['PassengerTitle'] = df['Name'].str.extract('([A-Za-z]+)', expand=False)
    
    title_replacements = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare',
        'Col': 'Rare', 'Don': 'Rare', 'Dr': 'Rare', 
        'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare', 
        'Jonkheer': 'Rare', 'Dona': 'Rare', 'Mlle': 'Miss', 
        'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['PassengerTitle'] = df['PassengerTitle'].replace(title_replacements)

create_features(titanic_data)
print(titanic_data.head())
print(titanic_data.describe())

# Data visualization
def plot_survival_distribution(df):
    plt.figure(figsize=(6, 4))
    df['Survived'].value_counts().plot(kind='bar', color=['salmon', 'lightgreen'])
    plt.title('Survival Distribution of Passengers')
    plt.xlabel('Survived (0: No, 1: Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()  # Display the figure

def plot_demographics(df):
    plt.figure(figsize=(10, 4))

    # Gender distribution
    plt.subplot(1, 2, 1)
    df['Sex'].value_counts().plot(kind='bar', color=['lightblue', 'pink'])
    plt.title('Passenger Gender Distribution')
    plt.xticks(rotation=0)
    plt.xlabel('Gender')
    plt.ylabel('Count')

    # Embarkation distribution
    plt.subplot(1, 2, 2)
    df['Embarked'].value_counts().plot(kind='bar', color=['gold', 'blue', 'red'])
    plt.title('Passenger Embarkation Distribution')
    plt.xlabel('Embarked')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()  # Display the figure

plot_survival_distribution(titanic_data)
plot_demographics(titanic_data)

# Missing values analysis
def analyze_missing_values(df):
    print(df.isnull().sum())
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    print(missing_percentage.to_frame('Missing Percentage').sort_values(by='Missing Percentage', ascending=False))

analyze_missing_values(titanic_data)

# Data preprocessing
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilyCount']
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Sex', 'Embarked', 'PassengerTitle', 'IsSingle']
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Prepare data for modeling
X = titanic_data[numeric_features + categorical_features]
y = titanic_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)

# Metrics calculation
def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred)

    print("Accuracy:", accuracy, f" - {accuracy * 100:.2f}%")
    print("Precision:", precision, f" - {precision * 100:.2f}%")
    print("Recall:", recall, f" - {recall * 100:.2f}%")
    print("F1 Score:", f1, f" - {f1 * 100:.2f}%")
    print("Classification Report:\n", classification_rep)

print_metrics(y_val, y_pred)

# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("Cross-validation Accuracy:", cv_scores.mean())
