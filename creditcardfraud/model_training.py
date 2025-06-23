import sys
import os
import warnings
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DependencyManager

deps = DependencyManager()

pd = deps.auto_import('pandas')
np = deps.auto_import('numpy')
sns = deps.auto_import('seaborn')
plt = deps.auto_import('matplotlib.pyplot')

train_test_split = deps.auto_import('sklearn.model_selection').train_test_split
StratifiedKFold = deps.auto_import('sklearn.model_selection').StratifiedKFold
GridSearchCV = deps.auto_import('sklearn.model_selection').GridSearchCV

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
joblib = deps.auto_import('joblib')
shap = deps.auto_import('shap')

warnings.filterwarnings('ignore')
np.random.seed(42)

# leitura dos dados
df = pd.read_csv('creditcard_fraud_dataset/creditcard.csv')

# criacao de novas features
df['log_amount'] = np.log1p(df['Amount'])
df['hour'] = (df['Time'] // 3600) % 24

# matriz de correlacao
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('matriz de correlacao')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

print('correlacao com "Class":')
print(corr['Class'].sort_values(ascending=False))

# separacao de features e alvo
X = df.drop(columns='Class')
y = df['Class']

# padronizacao de colunas numericas
num_cols = ['Amount', 'Time', 'log_amount', 'hour']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# divisao treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# balanceamento com smote
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# pipeline com gridsearch para xgboost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.05, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print('melhor roc auc no gridsearch:')
print(grid.best_score_)

# avaliacao no conjunto de teste
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]

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

# importancia de variaveis com shap
explainer = shap.Explainer(grid.best_estimator_.named_steps['clf'], X_train_res)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values, max_display=10)

# salvar modelo treinado
joblib.dump(grid.best_estimator_, 'best_model.pkl')

# salvar metricas
metrics = {
    'roc_auc': float(roc_auc),
    'average_precision': float(avg_prec),
    'confusion_matrix': matrix.tolist()
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
