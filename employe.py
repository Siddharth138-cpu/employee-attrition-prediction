import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("D:\\projects\\sid\\siddd.venv\\Employee-Attrition.csv")
print(df.columns.tolist())


# Identify categorical columns automatically (object or category types)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Create encoders dictionary
label_encoders = {}

# Encode all categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features/target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Encode target
target_encoder = label_encoders["Attrition"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save everything
joblib.dump(model, "employee_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model & encoders saved successfully!")
