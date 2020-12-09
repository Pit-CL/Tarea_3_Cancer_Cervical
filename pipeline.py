from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report

# Archivo que contiene los datos
file = '/home/rafaelfp/Dropbox/Postgrados/MDS/ML/ML2/Tarea_3/data/sobar-72.csv'

# Se crea el df en pandas
df = pd.read_csv(file)

# Se revisa el data set
head = df.head()
print('Un resumen del DF\n', head)

# Se revisa la info de la data para conocer las variables asociadas.
info = df.info()
print('Información de las variables\n', info)

# Describe para ver rápidamente el comportamiento de las variables.
describe = df.describe()
print('Descripción estadística\n', describe)

# Se revisa la distribución de los datos.
sns.distplot(df)
plt.show()

# Se revisa el balanceo de los datos de cáncer.
sns.countplot(x='ca_cervix', data=df)
plt.title('Distribución de las clases objetivo')
plt.show()

# Se revisa presencia de outliers.
plt.xticks(rotation=90)
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Separando el df
x = df.iloc[:, :19]
y = df.iloc[:, 19]

# Dividiendo el df en test y entrenamiento.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                    random_state=42)

# Creando los pipelines.
pipe_svm = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                     ('clf', svm.SVC(random_state=42))])

pipe_RF = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                    ('clf', RandomForestClassifier(random_state=42))])

pipe_grad = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                      ('clf', GradientBoostingClassifier(random_state=42))])

pipe_BRF = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                     ('clf', BalancedRandomForestClassifier(random_state=42))])

# Lista de pipelines
pipelines = [pipe_svm, pipe_RF, pipe_grad, pipe_BRF]

# Fit a los pipelines
for pipe in pipelines:
    pipe.fit(x_train, y_train)


# Diccionario de los pipelines
pipe_dict = {0: 'SVM', 1: 'RF', 2: 'GradienBoost', 3: 'BRF'}
print('======================================================================')

# Comparando accuracies.
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.5f' % (
        pipe_dict[idx], val.score(x_test, y_test)))

# Identificando el modelo con mejor accuracy en los datos de testeo.
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(pipelines):
    if val.score(x_test, y_test) > best_acc:
        best_acc = val.score(x_test, y_test)
        best_pipe = val
        best_clf = idx
print('======================================================================')
print('El modelo clasificador con mejor accuracy es: %s' % pipe_dict[best_clf])
print('======================================================================')

# Imprimiendo las matrices de confusion.
for idx, val in enumerate(pipelines):
    print('Matriz de confusión para', pipe_dict[idx], 'es:\n', confusion_matrix
          (y_test, val.predict(x_test)))

print('======================================================================')

# Imprimiendo los reportes de clasificacion.
for idx, val in enumerate(pipelines):
    print('Reporte de clasificacion para', pipe_dict[idx], 'es:\n',
          classification_report(y_test, val.predict(x_test)))

