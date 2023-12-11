# %%
# !pip install --upgrade pip
# !pip install pandas scikit-learn
# !pip install lightgbm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# from lightgbm import LGBMClassifier

# Data Preparation

# df = pd.read_csv("https://www.cpe.ku.ac.th/~cnc/customer_data.csv")
df = pd.read_csv("customer_data.csv")

df = df[~df["Segmentation"].isna()]

df["Segmentation"] = df["Segmentation"].map({"A": 0, "B": 1, "C": 2, "D": 3})

# Preprocessing

df_target = df["Segmentation"]
df = df.drop("Segmentation", axis=1)

categorical_cols = df.select_dtypes(include="object").columns.tolist()

encoders = dict()

for col in categorical_cols:
    series = df[col]
    label_encoder = LabelEncoder()
    df[col] = pd.Series(
        label_encoder.fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index,
    )
    encoders[col] = label_encoder

# for encoder in encoders:
#     print(encoder, encoders[encoder].classes_)

# Missing Value Analysis

df = df.ffill()

# ML

X = df.drop("ID", axis=1)

y = df_target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69
)

# params = {
#     "num_leaves": 75,
#     "max_depth": 6,
#     "learning_rate": 0.05,
#     "n_estimators": 850,
#     "min_child_samples": 76,
#     "subsample": 0.859830660090177,
#     "colsample_bytree": 0.7614632442258541,
#     "reg_alpha": 9.477998383389581,
#     "reg_lambda": 4.6427108580018315,
#     "random_state": 69,
# }
# model = LGBMClassifier(**params, verbose=-1)

params = {
    "n_estimators": 35,
    "max_depth": 12,
    "min_samples_split": 114,
    "min_samples_leaf": 6,
    "max_features": None,
    "random_state": 42,
}

model = RandomForestClassifier(**params)

model.fit(X_train, y_train)

print(params)
print(accuracy_score(y_test, model.predict(X_test)))
