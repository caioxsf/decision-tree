import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def arvore_decisao_emprestimos():
    # Carregar dataset traduzido
    df = pd.read_csv("emprestimos.csv")

    # Selecionar colunas relevantes
    atributos = [
        "dependentes", "educacao", "autonomo", "renda_anual", "valor_emprestimo",
        "prazo_emprestimo", "pontuacao_credito", "bens_residenciais",
        "bens_comerciais", "bens_luxo", "ativos_bancarios"
    ]
    X = df[atributos]
    y = df["aprovado"]

    # Identificar variáveis categóricas e numéricas
    categ = ["educacao", "autonomo"]
    numericos = [c for c in atributos if c not in categ]

    # Pré-processamento
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessador = ColumnTransformer([
        ("num", num_transformer, numericos),
        ("cat", cat_transformer, categ)
    ])

    # Modelo de árvore
    modelo = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=42)

    pipeline = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", modelo)
    ])

    # Divisão treino/teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Treinamento
    pipeline.fit(X_treino, y_treino)
    y_pred = pipeline.predict(X_teste)

    # Avaliação
    acc = accuracy_score(y_teste, y_pred)
    cm = confusion_matrix(y_teste, y_pred)
    print(f"\n📈 Acurácia: {acc:.3f}")
    print("📊 Matriz de Confusão:\n", cm)
    print("\n📋 Relatório de Classificação:\n", classification_report(y_teste, y_pred, digits=3))

    # Visualização da árvore
    X_treino_t = pipeline.named_steps["preprocessador"].transform(X_treino)
    cat_encoder = pipeline.named_steps["preprocessador"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categ)
    nomes_final = numericos + list(cat_names)

    plt.figure(figsize=(22, 12))
    plot_tree(
        modelo,
        feature_names=nomes_final,
        class_names=["Rejeitado", "Aprovado"],
        filled=True, rounded=True, fontsize=9
    )
    plt.title("Árvore de Decisão - Aprovação de Empréstimos", fontsize=16)
    plt.savefig("arvore_emprestimos.png", dpi=200, bbox_inches="tight")
    plt.show()

# Executar
arvore_decisao_emprestimos()
