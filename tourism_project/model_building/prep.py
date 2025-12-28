# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Andrew2505/Tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define features for numerical and categorical
numeric_features = ['Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'OwnCar', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus','Designation']

# Convert the specified columns in the DataFrame 'df' to 'category' dtype
for col_name in categorical_features:
    tourism[col_name] = tourism[col_name].astype("category")

# Define the target variable for the classification task
target = 'ProdTaken'

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism[numeric_features + categorical_features]

# Define target variable
y = tourism[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Andrew2505/Tourism-package-prediction",
        repo_type="dataset",
    )
