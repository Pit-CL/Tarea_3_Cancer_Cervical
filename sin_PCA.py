from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, \
    GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Archivo que contiene los datos
file = '/home/rafaelfp/Dropbox/Postgrados/MDS/ML/ML2/Tarea_3/data/sobar-72.csv'

# Creo el df en pandas
df = pd.read_csv(file)

# Separando el df entre variables dependientes e independientes.
x = df.iloc[:, :19]
y = df.iloc[:, 19]

print(x.shape)
print(y.shape)

# Dividiendo el df en test y entrenamiento.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                    random_state=45)

pipe_rl = Pipeline([('scl', StandardScaler()),
                    ('clf', LogisticRegression(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('clf', svm.SVC(random_state=42))])

pipe_RF = Pipeline([('scl', StandardScaler()),
                    ('clf', RandomForestClassifier(random_state=42))])

pipe_DT = Pipeline([('scl', StandardScaler()),
                    ('clf', DecisionTreeClassifier(random_state=42))])

pipe_ADA = Pipeline([('scl', StandardScaler()),
                     ('clf', AdaBoostClassifier(random_state=42))])

pipe_bagin = Pipeline([('scl', StandardScaler()),
                       ('clf', BaggingClassifier(random_state=42))])

pipe_grad = Pipeline([('scl', StandardScaler()),
                      ('clf', GradientBoostingClassifier(random_state=42))])

pipe_MLPC = Pipeline([('scl', StandardScaler()),
                      ('clf', MLPClassifier(random_state=42))])

# Lista de pipelines
pipelines = [pipe_rl, pipe_svm, pipe_RF, pipe_DT, pipe_ADA, pipe_bagin,
             pipe_grad, pipe_MLPC]

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(x_train, y_train)

# Diccionario de los pipelines
pipe_dict = {0: 'RL', 1: 'SVM', 2: 'RF', 3: 'DT', 4: 'ADA', 5: 'Bagin',
             6: 'Gradient', 7: 'MLCP'}

# Comparando accuracies.
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.5f' % (
        pipe_dict[idx], val.score(x_test, y_test)))