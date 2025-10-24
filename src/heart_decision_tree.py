import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def execute_decision_tree():

    df = pd.read_csv("./data/heart_disease_uci.csv")

    feature_cols = [
        "age", "sex", "cp", 
        "trestbps", "chol", "fbs", 
        "restecg", "thalch", "exang", 
        "oldpeak", "slope", "ca", "thal"
    ]

    x = df[feature_cols]
    y = (df["num"] > 0).astype(int)

    categorical_fetuares = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric_features = [c for c in feature_cols if c not in categorical_fetuares]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_fetuares)
        ]
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Acurácia: {acc:.3f}")
    print("Matriz de Confusão:\n", cm)
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred, digits=3))

    X_train_t = pipe.named_steps["preprocess"].transform(x_train)

    num_names = numeric_features
    cat_encoder = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_fetuares)
    feature_names = list(num_names) + list(cat_names)

    clf_plot = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        random_state=42
    )
    clf_plot.fit(X_train_t, y_train)

    importances = clf_plot.feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 10 importâncias de features:")
    print(imp_series.head(10))

    plt.figure(figsize=(22, 12))
    plot_tree(
        clf_plot,
        feature_names=feature_names,
        class_names=["No Disease (0)", "Disease (1)"],
        filled=True, rounded=True, fontsize=9
    )
    plt.title("Decision Tree - UCI Heart Disease")
    plt.savefig("heart_decision_tree.png", dpi=200, bbox_inches="tight")
    plt.show()