import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
import skops.io as sio
from huggingface_hub import HfApi, login
from kaggle_secrets import UserSecretsClient

warnings.filterwarnings('ignore')

# 1. CARREGAMENTO DOS DADOS
print("Carregando bases...")
reviews = pd.read_csv('/kaggle/input/datasets/olistbr/brazilian-ecommerce/olist_order_reviews_dataset.csv')
orders = pd.read_csv('/kaggle/input/datasets/olistbr/brazilian-ecommerce/olist_orders_dataset.csv')
customers = pd.read_csv('/kaggle/input/datasets/olistbr/brazilian-ecommerce/olist_customers_dataset.csv')

# 2. MERGE E LIMPEZA
df = reviews.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp']], on='order_id', how='left')
df = df.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='left')

df = df.dropna(subset=['review_comment_message']).copy()
df = df.drop_duplicates(subset=['review_comment_message'])

# Remove a nota 3 (ambígua) e criamos o alvo binário
df = df[df['review_score'] != 3].copy()
df['is_negative'] = df['review_score'].apply(lambda x: 1 if x <= 2 else 0)

# Ordenação temporal
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df = df.sort_values(by='order_purchase_timestamp').reset_index(drop=True)

# 3. DIVISÃO SEM VAZAMENTO (TEMPORAL + ISOLAMENTO DE CLIENTE)
split_idx = int(len(df) * 0.8)
train_val_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# Remove do teste qualquer cliente que já apareceu no passado
test_df = test_df[~test_df['customer_unique_id'].isin(train_val_df['customer_unique_id'])]

X_tv = train_val_df['review_comment_message']
y_tv = train_val_df['is_negative']

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.3, random_state=42, stratify=y_tv
)

print(f"Treino: {len(X_train)} | Val: {len(X_val)} | Teste: {len(test_df)}")

# 4. TREINO BASELINE E CALIBRAÇÃO
print("\nTreinando TF-IDF + Regressão Logística...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)

print("Calibrando probabilidades (Regressão Isotônica)...")
calibrated_clf = CalibratedClassifierCV(pipeline, method='isotonic', cv='prefit')
calibrated_clf.fit(X_val, y_val) 

# 5. BUSCA DO LIMIAR PARA 95% DE PRECISÃO
probs_val = calibrated_clf.predict_proba(X_val)[:, 1]
brier = brier_score_loss(y_val, probs_val)
print(f"Brier Score na Validação: {brier:.4f}")

thresholds = np.linspace(0.5, 0.99, 100)
optimal_threshold = 0.99

for t in thresholds:
    preds = (probs_val >= t).astype(int)
    prec = precision_score(y_val, preds, zero_division=0)
    if prec >= 0.95:
        optimal_threshold = t
        break

print(f"Limiar escolhido para >=95% precisão: {optimal_threshold:.4f}")

# 6. TESTE FINAL (FUTURO INÉDITO)
X_test = test_df['review_comment_message']
y_test = test_df['is_negative']

probs_test = calibrated_clf.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= optimal_threshold).astype(int)

print("\n--- RESULTADOS FINAIS NO TESTE ---")
print(classification_report(y_test, preds_test, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_test, probs_test):.4f}")

# Salva o modelo calibrado que ainda está na memória do notebook para um arquivo físico
sio.dump(calibrated_clf, "modelo_olist_tfidf_calibrado.skops")

print("Arquivo .skops gerado com sucesso! Agora você pode rodar o bloco de upload.")

# 1. Autenticação via Kaggle Secrets
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

# 2. Defina o nome do seu repositório no Hugging Face
repo_id = "DaviCM/olist-review-triage-calibrated"

# 3. Criação do repositório e upload
api = HfApi()
print(f"Criando repositório {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print("Fazendo upload do modelo .skops...")
api.upload_file(
    path_or_fileobj="modelo_olist_tfidf_calibrado.skops",
    path_in_repo="modelo_olist_tfidf_calibrado.skops",
    repo_id=repo_id,
    repo_type="model"
)
print("✅ Deploy concluído com sucesso!")