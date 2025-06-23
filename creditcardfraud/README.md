# creditcardfraud

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/badge/status-complete-success)
![Dataset](https://img.shields.io/badge/dataset-Kaggle-blue)

Detecção de fraudes em transações com cartão de crédito utilizando técnicas de Machine Learning supervisionado e balanceamento de classes. O projeto foi construído com foco em avaliação de múltiplos modelos, balanceamento com SMOTE, análise de correlação e padronização de variáveis.

---

## Estrutura do Projeto

```bash
creditcardfraud/
│
├── creditcard_fraud/                 # Diretório com o dataset original
│   └── creditcard.csv                # Base de dados de transações
│
├── __pycache__/                      # Cache do Python
│
├── best_model.pkl                    # Modelo treinado e serializado
├── correlation_matrix.png           # Matriz de correlação das variáveis
├── creditcardfraud.zip              # Arquivo ZIP original do Kaggle
├── metrics.json                      # Métricas de avaliação final
│
├── download_db.py                   # Script para baixar e extrair o dataset
├── evaluation_model.py              # Avaliação dos modelos de ML
├── utils.py                         # Gerenciador automático de dependências
└── README.md                        # Este arquivo

Dataset
O projeto utiliza o dataset de fraude com cartão de crédito disponibilizado no Kaggle, contendo transações europeias feitas por cartões em setembro de 2013.

Registros: 284.807

Fraudes: 492 (0.17%)

Atributos: 30 (anônimos via PCA + Time e Amount)

Target: Class (0 = legítima, 1 = fraude)

Como Executar
1. Clone o repositório
bash
Copiar
Editar
git clone https://github.com/seu-usuario/creditcardfraud.git
cd creditcardfraud
2. Configure sua chave da API do Kaggle
Salve o kaggle.json (chave da API) no diretório:

bash
Copiar
Editar
~/.kaggle/kaggle.json
3. Baixe e extraia o dataset
bash
Copiar
Editar
python download_db.py
4. Execute o pipeline de avaliação
bash
Copiar
Editar
python evaluation_model.py
Isso irá:

Carregar o dataset

Gerar a matriz de correlação

Padronizar variáveis

Balancear com SMOTE

Avaliar cinco modelos:

Logistic Regression

Random Forest

XGBoost

Naive Bayes

SVM (RBF)

Gerenciamento de Dependências
Este projeto utiliza um gerenciador inteligente de dependências via utils.py. Pacotes ausentes serão instalados automaticamente.

Você pode instalar manualmente todos os requisitos com:

bash
Copiar
Editar
pip install -r requirements.txt
Um requirements.txt pode ser gerado a partir do log de dependências importadas.

Resultados
As métricas de desempenho incluem:

ROC AUC

Average Precision

Classification Report

Confusion Matrix

Essas métricas são salvas no arquivo metrics.json, e um resumo ordenado por ROC AUC é impresso no terminal.

📌 Principais Destaques Técnicos
📐 Padronização de Features (StandardScaler)

🔍 Análise de Correlação (seaborn)

⚖️ Balanceamento com SMOTE (imblearn)

🔁 Cross-validation

🧠 Comparação de Modelos Supervisionados

📊 Visualização da Matriz de Correlação

