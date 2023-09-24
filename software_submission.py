import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

df_dev = pd.read_csv('fall_project_dataset/development.csv', index_col=0)
df_eval = pd.read_csv('fall_project_dataset/evaluation.csv', index_col=0)

df = pd.concat([df_dev, df_eval])

# Reduce the cardinality of the OCCP column 529 -> 25
# Create a dictionary from the OCCP code to the text representation
reader = csv.reader(open('produced_documents/occp_to_string.csv', 'r'), delimiter=';')
next(reader, None) # Skip the headers

occp_to_string = {}

for row in reader:
   k, v = row
   k = int(k)
   occp_to_string[k] = v

# Map the OCCP column to its text values
df['OCCP'] = df['OCCP'].map(occp_to_string)

# Keep only the first 3 characters 
df['OCCP'] = df['OCCP'].apply(lambda occp : occp[0:3])

# Group the countries of POBP column by continent
# 219 -> 6
df['POBP'] = pd.cut(df['POBP'], bins=[0,1,100,200,300,400,500], right=False, labels=['N/A', 'USA', 'Europe', 'Asia', 'Americas', 'Oceania'], include_lowest=True)

# Group the countries of MIGSP column by continent
# 96 -> 6
df['MIGSP'] = pd.cut(df['MIGSP'], bins=[0,1,100,200,300,400,500], right=False, labels=['N/A', 'USA', 'Europe', 'Asia', 'Americas', 'Oceania'], include_lowest=True)

# Group fine grained education categories together
# 24 -> 10
df['SCHL'] = pd.cut(df['SCHL'], bins=[0,1,8,10,15,19,20,21,22,23,24], right=True, labels=['No', 'Low', 'Primary', 'Junior High', 'High', 'Associate', 'Bachelor', 'Master', 'Professional', 'PhD'], include_lowest=True)

reader = csv.reader(open('produced_documents/JWDP.csv', 'r'))
next(reader, None) # Skip the headers

jwdp_begin = {}
jwdp_end = {}

for row in reader:
   k, b, e = row
   k = int(k)
   jwdp_begin[k] = int(b)
   jwdp_end[k] = int(e)

## Get the minutes range from JWAP and JWDP and calculate the possible range of JWMNP

reader = csv.reader(open('produced_documents/JWAP.csv', 'r'))
next(reader, None) # Skip the headers

jwap_begin = {}
jwap_end = {}

for row in reader:
   k, b, e = row
   k = int(k)
   jwap_begin[k] = int(b)
   jwap_end[k] = int(e)

# Map the JWDP column to extract minimum and maximum departure time in minutes
df['JWDP_B'] = df['JWDP'].map(jwdp_begin)
df['JWDP_E'] = df['JWDP'].map(jwdp_end)

# Map the JWAP column to extract minimum and maximum arrival time in minutes
df['JWAP_B'] = df['JWAP'].map(jwap_begin)
df['JWAP_E'] = df['JWAP'].map(jwap_end)

# Add two columns for the expected JWMNP range
df['JWMNP_B'] = df['JWAP_B'] - df['JWDP_B']
df['JWMNP_E'] = df['JWAP_E'] - df['JWDP_E']
df['JWMNP_B_E'] = df['JWAP_E'] - df['JWDP_B']
df['JWMNP_E_B'] = df['JWAP_B'] - df['JWDP_E']
df['JWMNP_A'] = (df['JWAP_E'] - df['JWDP_B']) / 2

# Drop high correlation columns and all the JWDP, JWAP columns
df = df.drop(columns=['MIG', 'PAOC', 'FER', 'VPS', 'JWDP', 'JWAP'])

# Keep only the 5 most frequent values in the LANP column
top5 = df['LANP'].value_counts().head(5).index

# In case the language is not in the list the assigned value is NaN
# 121 -> 5
df['LANP'] = df['LANP'].where(df['LANP'].isin(top5))

numeric_features = ['WKHP', 'PINCP', 'JWMNP_B', 'JWMNP_E', 'JWMNP_B_E', 'JWMNP_E_B', 'JWMNP_A']
categorical_features = ['JWDP_B', 'JWDP_E', 'JWAP_B', 'JWAP_E', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'SEX', 'RAC1P', 'HICOV', 'LANP', 'PUBCOV', 'DEAR', 'MIGSP', 'ENG', 'OC', 'FDEYEP', 'MIL']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop="if_binary")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

df_dev = df[df["JWMNP"].notna()]
df_eval = df[df["JWMNP"].isna()]

X = df_dev.drop(columns=["JWMNP"])
y = df_dev["JWMNP"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', GradientBoostingRegressor(learning_rate=0.2, n_estimators=75, criterion="friedman_mse", max_depth=6, random_state=42))])

pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regressor', GradientBoostingRegressor(learning_rate=0.2, n_estimators=75, criterion="friedman_mse", max_depth=6, random_state=42))])

pipe.fit(X, y)

y_pred = pipe.predict(df_eval)

data = list(zip(df_eval.index, y_pred))

# Save the data to a CSV file
with open('test_submission.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Id', 'Predicted'])  # Header row
    csvwriter.writerows(data)