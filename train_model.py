import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib 

#Load the dataset
data = "Diabetes_cleaned_data.csv"
df = pd.read_csv(data)

#Split the dataset into X and y, but first we remove the "Unnamed" column

if 'Unnamed: 0' in df.columns:
    df=df.drop('Unnamed: 0',axis=1)
    print("Unamed found")

#X = df.drop('Outcome',axis = 1)

#X.info()

X = df.drop('Outcome',axis = 1)
y = df['Outcome'] 

#Train the model
model = RandomForestClassifier()
model.fit(X,y)

#Save the model
joblib.dump(model,'diabetes_app.pkl')
print("The model is saved successfully")