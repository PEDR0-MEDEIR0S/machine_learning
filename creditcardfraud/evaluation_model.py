import sys
import os
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DependencyManager

deps = DependencyManager()

pd = deps.auto_import('pandas')
sns = deps.auto_import('seaborn')
plt = deps.auto_import('matplotlib.pyplot')

train_test_split = deps.auto_import('sklearn.model_selection').train_test_split
cross_val_score = deps.auto_import('sklearn.model_selection').cross_val_score

StandardScaler = deps.auto_import('sklearn.preprocessing').StandardScaler
Pipeline = deps.auto_import('sklearn.pipeline').Pipeline

classification_report = deps.auto_import('sklearn.metrics').classification_report
roc_auc_score = deps.auto_import('sklearn.metrics').roc_auc_score
average_precision_score = deps.auto_import('sklearn.metrics').average_precision_score
confusion_matrix = deps.auto_import('sklearn.metrics').confusion_matrix

LogisticRegression = deps.auto_import('sklearn.linear_model').LogisticRegression
RandomForestClassifier = deps.auto_import('sklearn.ensemble').RandomForestClassifier
XGBClassifier = deps.auto_import('xgboost').XGBClassifier
GaussianNB = deps.auto_import('sklearn.naive_bayes').GaussianNB
SVC = deps.auto_import('sklearn.svm').SVC

SMOTE = deps.auto_import('imblearn.over_sampling').SMOTE

warnings.filterwarnings('ignore')
np.random.seed(42)

# leitura dos dados
df = pd.read_csv('creditcard_fraud_dataset/creditcard.csv')

# matriz de correlacao
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('matriz de correlacao')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# exibicao da correlacao com a variavel alvo
print('correlacao com "Class":')
print(corr['Class'].sort_values(ascending=False))

# separacao de features e alvo
X = df.drop(columns='Class')
y = df['Class']

# padronizacao das variaveis numericas
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# divisao treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# aplicacao do smote para balancear a base
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# definicao dos modelos
models = {
    'logistic regression': LogisticRegression(),
    'random forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'naive bayes': GaussianNB(),
    'svm rbf': SVC(probability=True)
}

# dicionario para salvar resultados
results = []

# loop de treino e avaliacao
for name, model in models.items():
    print(f'\ntreinando: {name}')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    
    pipeline.fit(X_train_res, y_train_res)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    matrix = confusion_matrix(y_test, y_pred)

    print('classification report:')
    print(report)
    print(f'roc auc: {roc_auc:.4f}')
    print(f'average precision: {avg_prec:.4f}')
    print('confusion matrix:')
    print(matrix)

    results.append({
        'model': name,
        'roc_auc': roc_auc,
        'avg_precision': avg_prec
    })

# exibicao final dos resultados
results_df = pd.DataFrame(results).sort_values(by='roc_auc', ascending=False)
print('\nresumo dos modelos:')
print(results_df.reset_index(drop=True))
