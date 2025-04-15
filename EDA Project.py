import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset to examine its structure
file_path = "D:\Sem 4\INT 375(Python)\CA2\Crime_Data_from_2020_to_Present.csv"
df = pd.read_csv(file_path)

df.info()


crime_df = df.copy()

# Drop rows where essential values are missing
essential_cols = ['DR_NO', 'Date Rptd', 'TIME OCC', 'AREA', 'AREA NAME', 'Crm Cd', 'Crm Cd Desc', 'Vict Age', 'Vict Sex']
crime_df.dropna(subset=essential_cols, inplace=True)

# Fill missing values
crime_df['Vict Sex'].fillna(crime_df['Vict Sex'].mode()[0], inplace=True)
crime_df['Premis Desc'].fillna(crime_df['Premis Desc'].mode()[0], inplace=True)
crime_df['Premis Cd'].fillna(crime_df['Premis Cd'].mode()[0], inplace=True)
crime_df['Weapon Used Cd'].fillna(crime_df['Weapon Used Cd'].mode()[0], inplace=True)
crime_df['Weapon Desc'].fillna(crime_df['Weapon Desc'].mode()[0], inplace=True)
crime_df['Mocodes'].fillna('UNKNOWN', inplace=True)
crime_df['Cross Street'].fillna('UNKNOWN', inplace=True)

# Convert columns to proper types
crime_df['DR_NO'] = crime_df['DR_NO'].astype(int)
crime_df['TIME OCC'] = crime_df['TIME OCC'].astype(int)
crime_df['Vict Age'] = crime_df['Vict Age'].astype(int)
crime_df['Premis Cd'] = crime_df['Premis Cd'].astype(int)
crime_df['Weapon Used Cd'] = crime_df['Weapon Used Cd'].astype(int)

# Convert date to datetime
crime_df['Date Rptd'] = pd.to_datetime(crime_df['Date Rptd'], errors='coerce')
crime_df.dropna(subset=['Date Rptd'], inplace=True)

# Reset index
crime_df.reset_index(drop=True, inplace=True)


crime_df.head()


# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Top 10 most common crime types
top_crimes = crime_df['Crm Cd Desc'].value_counts().head(10)
plt.figure()
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette='viridis')
plt.title("Top 10 Most Common Crime Types")
plt.xlabel("Number of Incidents")
plt.ylabel("Crime Type")
plt.tight_layout()
plt.show()

# 2. Pie chart for Victim Sex
plt.figure()
crime_df['Vict Sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'lightcoral'])
plt.title("Victim Gender Distribution")
plt.ylabel('')
plt.tight_layout()
plt.show()

# 3. Histogram for Victim Age
plt.figure()
sns.histplot(crime_df['Vict Age'], bins=30, kde=True, color='salmon')
plt.title("Victim Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 4. Boxplot for Victim Age (Outlier detection)
plt.figure()
sns.boxplot(x=crime_df['Vict Age'], color='orange')
plt.title("Boxplot of Victim Age")
plt.xlabel("Age")
plt.tight_layout()
plt.show()

# 5. Lineplot - Crime reports over time
crime_df['Year'] = crime_df['Date Rptd'].dt.year
yearly_crimes = crime_df['Year'].value_counts().sort_index()
plt.figure()
sns.lineplot(x=yearly_crimes.index, y=yearly_crimes.values, marker='o', color='purple')
plt.title("Crimes Reported Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.tight_layout()
plt.show()

# 6. Correlation heatmap (select numerical features only)
numeric_cols = crime_df.select_dtypes(include=['int64', 'float64']).drop(columns=['DR_NO']).copy()
plt.figure()
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()


# Select features and target
features = ['AREA', 'TIME OCC', 'Crm Cd', 'Premis Cd']
target = 'Vict Age'

X = crime_df[features]
y = crime_df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

(mae, rmse, r2)
