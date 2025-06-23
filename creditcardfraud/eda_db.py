import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import DependencyManager

deps = DependencyManager()

kaggle = deps.auto_import('kaggle')
pd = deps.auto_import('pandas')
openml = deps.auto_import('openml')
sns = deps.auto_import('seaborn')
plt = deps.auto_import('matplotlib', 'matplotlib.pyplot')
sm = deps.auto_import('statsmodels', 'statsmodels.api')
Logit = sm.Logit
add_constant = sm.add_constant

path_local = 'creditcard_fraud_dataset/creditcard.csv'
db = '1597'

if os.path.exists(path_local):
    df = pd.read_csv(path_local)
    print('ds ja baixado')
else:
    df, *_ = openml.datasets.get_dataset(db).get_data(dataset_format='dataframe')
    print('usando openml')

print(f'visualizar no openml.org: openml.org/d/{db}')
print('head:')
print(df.head())
print('describe:')
print(df.describe())
print('nan:')
print(df.isnull().sum())
print('class:')
print(df['Class'].value_counts(normalize=True))
print(sm.tools.tools.add_constant(df.drop(columns='Class')).describe())

sns.countplot(data=df, x='Class')
plt.title('distribuicao (0 = normal, 1 = fraude)')
plt.tight_layout()
plt.show()

features = ['V1', 'V2', 'V3']
X = add_constant(df[features])
y = df['Class']

model = Logit(y, X).fit()
print(model.summary())
