import pandas as pd
import numpy as np

print("#----------------------------------------------------------#")
print("                          Data Set                          ")
print("#----------------------------------------------------------#")
data = pd.DataFrame({
    'age': [25, np.nan, 35, 40, 29, 60, 120],
    'income': [5000, 60000, 55000, np.nan, 45000, 1000000, 30000],
    'gender': ['male', 'female', 'female', np.nan, 'male', 'male', 'female'],
    'signup_date': pd.to_datetime(['2021-01-05', '2021-01-10', '2021-01-12', 
                                   '2021-02-01', '2021-02-25', '2021-03-10', 
                                   '2021-03-15'])
})
print(data)

print("#----------------------------------------------------------#")
print("                         Imputation                         ")
print("#----------------------------------------------------------#")
# Numerical imputation
data['age'] = data['age'].fillna(data['age'].median())
data['income'] = data['income'].fillna(data['income'].mean())

# Categorical imputation
most_common_gender = data['gender'].value_counts().idxmax()
data['gender'] = data['gender'].fillna(most_common_gender)
print(data)

print("#----------------------------------------------------------#")
print("                      Outlier Management                    ")
print("#----------------------------------------------------------#")
# Remove outliers in the 'income' column based on standard deviation
income_mean = data['income'].mean()
income_std = data['income'].std()
data = data[(data['income'] > income_mean - 2 * income_std) & (data['income'] < income_mean + 2 * income_std)]
print(data)

print("#----------------------------------------------------------#")
print("                       One-hot Encoding                     ")
print("#----------------------------------------------------------#")
data = pd.get_dummies(data, columns=['gender'], drop_first=True)
print(data)

print("#----------------------------------------------------------#")
print("                       Log Transform                        ")
print("#----------------------------------------------------------#")
# Apply the log transform to the 'income' column
data['log_income'] = np.log(data['income'] + 1)
print(data)

print("#----------------------------------------------------------#")
print("                          Scalling                          ")
print("#----------------------------------------------------------#")
data['age_norm'] = (data['age'] - data['age'].min()) / (data['age'].max() - data['age'].min())
# Standardization
data['age_std'] = (data['age'] - data['age'].mean()) / data['age'].std()
# table with column: before and after scaling
data_scaled = pd.DataFrame({'age_before': data['age'], 'age_norm': data['age_norm'], 'age_std': data['age_std']})
print(data_scaled)

print("#---------------------------------------------------------------------------------------------------------------#")
print("                                                   Manipulation                                        ")
print("#---------------------------------------------------------------------------------------------------------------#")
data['signup_day'] = data['signup_date'].dt.day
data['signup_month'] = data['signup_date'].dt.month
data['signup_weekday'] = data['signup_date'].dt.weekday
data['is_weekend'] = data['signup_weekday'].apply(lambda x: 1 if x >= 5 else 0)
print(data)
