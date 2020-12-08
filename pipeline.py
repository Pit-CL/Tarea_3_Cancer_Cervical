from sklearn.model_selection import train_test_split
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Archivo que contiene los datos
file = '/home/rafaelfp/Dropbox/Postgrados/MDS/ML/ML2/Tarea_3/data/sobar-72.csv'

# Creo el df en pandas
df = pd.read_csv(file)

# Reviso el balanceo de los datos de cáncer.
# No es un df muy balanceado por lo que el accuracy por sí solo no será el
# mejor indicador de la calidad del modelo.
# Debo revisar precisión, matriz de confusión, recall y f-1.
# Quizás DT o RF es el mejor para temas relacionados a clases no balanceadas ya
# que trabaja aprendiendo por jerarquía por lo que ambas clases son alcanzadas.
sns.countplot(x='ca_cervix', data=df)
plt.title('Distribución de las clases objetivo')
plt.show()

# Reviso presencia de outliers.}
plt.xticks(rotation=90)
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Separando el df entre variables dependientes e independientes.
x = df.iloc[:, :19]
y = df.iloc[:, 19]

# Dividiendo el df en test y entrenamiento.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                    random_state=42)

# Creando los pipes utilizando las 4 variables cuya varianza explicada es
# la más representativa. Esta representatividad está compuesta
#  por con un 29%, con un 19%, con un 11% y con un 9%. Con esto se completa
#  un 70% aproximado de representatividad de las variables.
# TODO: completar el nombre de las variables más representativas.
# TODO: Explicar porque se usa StandardScaler.
pipe_rl = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                    ('clf', LogisticRegression(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                     ('clf', svm.SVC(random_state=42))])

pipe_RF = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                    ('clf', RandomForestClassifier(random_state=42))])

pipe_DT = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                    ('clf', DecisionTreeClassifier(random_state=42))])

pipe_ADA = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                     ('clf', AdaBoostClassifier(random_state=42))])

pipe_bagin = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                       ('clf', BaggingClassifier(random_state=42))])

pipe_grad = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                      ('clf', GradientBoostingClassifier(random_state=42))])

pipe_MLCP = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=4)),
                      ('clf', MLPClassifier(random_state=42, max_iter=3000))])

# Lista de pipelines
pipelines = [pipe_rl, pipe_svm, pipe_RF, pipe_DT, pipe_ADA, pipe_bagin,
             pipe_grad, pipe_MLCP]

# Fit a los pipelines
for pipe in pipelines:
    pipe.fit(x_train, y_train)

# Diccionario de los pipelines
pipe_dict = {0: 'RL', 1: 'SVM', 2: 'RF', 3: 'DT', 4: 'ADA', 5: 'Bagin',
             6: 'Gradient', 7: 'MLCP'}

# Comparando accuracies.
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.5f' % (
        pipe_dict[idx], val.score(x_test, y_test)))

# Identificar el modelo con mejor accuracy en los datos de testeo.
best_acc = 0.0
best_clf = 0
best_pipe = ''
for idx, val in enumerate(pipelines):
    if val.score(x_test, y_test) > best_acc:
        best_acc = val.score(x_test, y_test)
        best_pipe = val
        best_clf = idx
print('----------------------------------------------------------------------')
print('El modelo clasificador con mejor accuracy es: %s' % pipe_dict[best_clf])

# TODO: Al ejecutar el script nos damos cuenta que basados sólo en el accuracy
# TODO: es poco probable determinar si RL es el mejor modelo ya que la data
# TODO: está desbalanceada.
# TODO: incluir matrices de confusión.
# TODO: Para revisar la importancia de las variables debo hacer PCA
# TODO: Ver si está balanceado o no.
# TODo: Calcular el y_pred para imprimir las matrices de confusión.
# TODO: Como random forest me entrega las variables importantes podría
# TODO: seleccionar que variables entran
# TODO: en 4 variables tengo el 70%

# for idx, val in enumerate(pipelines):
#     print('%s pipeline test accuracy: %.5f' % (
#         pipe_dict[idx], confusion_matrix(y_test, predict_proba)))
