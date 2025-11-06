import pandas as pd

# Lê o arquivo original baixado do Kaggle
df = pd.read_csv("loan_approval_dataset.csv")

# 🔹 Traduz nomes das colunas para português
df.rename(columns={
    "loan_id": "id_emprestimo",
    " no_of_dependents": "dependentes",
    " education": "educacao",
    " self_employed": "autonomo",
    " income_annum": "renda_anual",
    " loan_amount": "valor_emprestimo",
    " loan_term": "prazo_emprestimo",
    " cibil_score": "pontuacao_credito",
    " residential_assets_value": "bens_residenciais",
    " commercial_assets_value": "bens_comerciais",
    " luxury_assets_value": "bens_luxo",
    " bank_asset_value": "ativos_bancarios",
    " loan_status": "aprovado"
}, inplace=True)

# 🔹 Traduz valores categóricos
df["educacao"] = df["educacao"].replace({
    " Graduate": "Graduado",
    " Not Graduate": "Não Graduado"
})
df["autonomo"] = df["autonomo"].replace({
    " Yes": "Sim",
    " No": "Não"
})
df["aprovado"] = df["aprovado"].replace({
    " Approved": 1,
    " Rejected": 0
})

# 🔹 Salva o novo dataset traduzido
df.to_csv("emprestimos.csv", index=False, encoding="utf-8-sig")

print("✅ Dataset traduzido salvo com sucesso como 'emprestimos.csv'!")
