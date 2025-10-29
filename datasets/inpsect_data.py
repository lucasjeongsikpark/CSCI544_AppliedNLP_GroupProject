import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/ketan.joshi/USC/CSCI544/group-project/datasets/medredqa_train.csv")

# Display the column headers
print(df.columns.tolist())  # Show the list of column headers

# Display the first few rows and the column types
print(df.head(10))  # Show the first 10 rows
print(df.dtypes)     # Show the data types of each column