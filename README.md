# Triagem Automática de Avaliações com Garantia de Precisão (Olist)

Pipeline de Machine Learning orientado a produto para classificação de avaliações de e-commerce. O grande diferencial deste projeto não é forçar o modelo a prever 100% dos casos, mas sim **saber quando se abster**: o sistema decide autonomamente os casos seguros (elogios ou reclamações óbvias) e envia o restante (textos ambíguos, mistos ou irônicos) para revisão humana.

## Links do Projeto
- **Notebook de Análise e Execução (Kaggle):** https://www.kaggle.com/code/davicmaciel/brasil-olist
- **Modelo Empacotado & Model Card (Hugging Face):** https://huggingface.co/DaviCM/olist-review-triage-calibrated

---

## Arquitetura e Engenharia de Confiabilidade
Para garantir que o modelo funcione no mundo real sem degradação de performance, o pipeline implementa três defesas rígidas:
1. **Prevenção de Vazamento (Zero Leakage):** Divisão temporal estrita (treino no passado, teste no futuro) combinada com um `GroupShuffleSplit` e filtragem para garantir que nenhum cliente do teste tenha aparecido no treino/validação.
2. **Calibração Fora-da-Dobra (Out-of-Fold):** O modelo base (TF-IDF + Regressão Logística) é calibrado via Regressão Isotônica utilizando previsões K-Fold, evitando que o calibrador decore os dados e gere probabilidades falsamente confiantes.
3. **Limiares por Limite Inferior de Wilson:** O corte de decisão não é 0.5. O sistema busca a probabilidade exata onde o limite inferior do intervalo de confiança de Wilson atinge a meta de **95% de precisão**, punindo amostras pequenas e ruídos.

---

## Comparativo de Modelos (Teste Cego Temporal)
Avaliamos duas abordagens de vetorização sob o mesmo protocolo rigoroso com ~6,6 mil avaliações no teste futuro. O **Modelo A (Palavras)** foi o escolhido para produção devido à superioridade na precisão da classe minoritária (reclamações) e melhor calibração.

| Métrica no Teste Cego | Modelo A (TF-IDF Palavras) | Modelo B (TF-IDF Caracteres n: 2-5) | Baseline (Sem Abstenção) |
| :--- | :--- | :--- | :--- |
| **Cobertura Global** | **74,21%** | 74,48% | 100,0% |
| **Precisão Global (IC 95%)** | **96,96%** (96,4%–97,4%) | 96,42% (95,8%–96,9%) | 84,1% |
| **Cobertura da Classe Negativa** | 3,47% | 3,27% | - |
| **Precisão em Reclamações (Negativas)** | **98,67%** (Corte: 0.96) | 98,11% (Corte: 0.96) | - |
| **Erro Esperado de Calibração (ECE)** | **0.0142** | 0.0146 | - |

---

## Análise de Erros Qualitativa
A abstenção dupla provou-se vital ao analisar os casos em que o modelo recusa a automação e aciona o humano (faixa de incerteza):
* **Textos Mistos:** *"O produto é excelente, mas a transportadora amassou a caixa."* (Sentimentos conflitantes reduzem a probabilidade, acionando a revisão).
* **Ironia Curta:** *"Super rápido, demorou só dois meses."* (O vocabulário positivo induz o erro, mas a incerteza gerada impede a automação incorreta).

---

## Limitações do Projeto
* **Viés Temporal e de Domínio:** Dados de 2016–2018 do marketplace Olist. O modelo não mapeia neologismos logísticos atuais (ex: "remessa conforme").
* **Proxy de Rótulo:** A nota do usuário (1 e 2) foi utilizada como proxy para "reclamação textual", o que introduz ruído de notas baixas sem comentários úteis.
* **Deduplicação Restritiva:** A remoção de textos exatos elimina respostas curtas comuns de satisfação (ex: "Ótimo"), alterando a distribuição real da plataforma e subestimando levemente a cobertura verdadeira.
* **Escopo:** Este é um repositório de prova de conceito e portfólio; o modelo não está em ambiente de produção comercial (Licença CC BY-NC-SA 4.0 herdada dos dados).

---

## Diagnóstico Visual do Modelo Campeão

<p align="center">
  <img src="docs/diagnostico_modelo.png" alt="Diagnóstico do Modelo" width="800"/>
</p>

##  Estrutura do Repositório
```text
├── src/
│   └── preparo_e_treino.py   # Motor principal (validação, calibração, Wilson e abstenção)
├── docs/                     # Diagramas e relatórios de suporte
├── README.md                 # Documentação do projeto
└── requirements.txt          # Dependências do projeto