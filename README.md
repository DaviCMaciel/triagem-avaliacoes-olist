# Triagem de Avaliações Olist - Modelo Calibrado

Pipeline de Machine Learning para classificação de avaliações de e-commerce com garantia de precisão (abstenção de incertezas).

## Estrutura do Projeto
- `src/preparo_e_treino.py`: Script completo de extração temporal, prevenção de data leakage (isolamento por `customer_unique_id`), vetorização TF-IDF, treinamento do baseline logístico e calibração via Regressão Isotônica.
- O modelo final foi exportado de forma segura utilizando `skops`.

## Links
- **Notebook de Análise:** [Insira seu link do Kaggle]
- **Modelo Empacotado e Model Card:** [Insira seu link do Hugging Face]