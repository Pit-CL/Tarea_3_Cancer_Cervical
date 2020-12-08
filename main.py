# Importando librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Archivo que contiene los datos
file = '/home/rafaelfp/Dropbox/Postgrados/MDS/ML/ML2/Tarea_3/data/sobar-72.csv'

# Creo el df en pandas
df = pd.read_csv(file)

# Obteniendo información del df.
# Podemos notar que no hay valores nulos.
df.info()

# De todas maneras lo corroboramos
df.isnull().sum()

# Haremos una descripción de los datos.
# Se puede notar la posibilidad de la existencia de outliers.
df.describe()

# Graficaremos para visualizar la información.
# Podemos notar que no hay ua mayor presencia de outliers.
boxplot = sns.boxplot(df)
plt.show()

# Existe una relativa normalidad en los datos.
distplot = sns.distplot(df)
plt.show()

# Gráfico de correlación
# Se puede observar en el centro una alta correlación al igual que en la
# esquina inferior derecha.
f, ax = plt.subplots(figsize=(10, 8))

corr = df.corr()
heatmap = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                      cmap=sns.diverging_palette(10, 10, as_cmap=True),
                      square=True,
                      ax=ax)
plt.show()

# Separando el df entre variables dependientes e independientes.
x = df.iloc[:, :19]
y = df.iloc[:, 19]

print(x.shape)
print(y.shape)

# Dividiendo el df en test y entrenamiento.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                    random_state=45)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Estandarizando.
# Creando MinMax Scaler.
mm = MinMaxScaler()

# Alimentando la variable independiente.
x_train = mm.fit_transform(x_train)
x_test = mm.fit_transform(x_test)

# Regresión logística.
# Creando el modelo
model_log = LogisticRegression()

# Alimentando los datos de entrenamiento.
model_log.fit(x_train, y_train)

# prediciendo el resultado del test set.
y_pred = model_log.predict(x_test)

# Calculando accuracies.
print("Accuracy de entrenamiento:", model_log.score(x_train, y_train))
print("Accuracy de testeo:", model_log.score(x_test, y_test))

# Reporte de clasificación.
print(classification_report(y_test, y_pred))

# Matriz de confusión.
print(confusion_matrix(y_test, y_pred))

# Support Vector Machine
# Creando el modelo
model_SVC = SVC()

# Alimentando la data de entrenamiento.
model_SVC.fit(x_train, y_train)

# Prediciendo
y_pred = model_SVC.predict(x_test)

# Calculando los accuracies
print("Accuracy de entrenamiento :", model_SVC.score(x_train, y_train))
print("Accuracy de testeo :", model_SVC.score(x_test, y_test))

# Reporte de clasificación
print(classification_report(y_test, y_pred))

# Matriz de confusión
print(confusion_matrix(y_test, y_pred))

# Random Forest
# Creando el modelo.
model_Random = RandomForestClassifier()

# Alimentando los datos de entramiento.
model_Random.fit(x_train, y_train)

# Prediciendo.
y_pred = model_Random.predict(x_test)

# Calculando las accuracies.
print("Accuracy de entrenamiento :", model_Random.score(x_train, y_train))
print("Accuracy de testeo :", model_Random.score(x_test, y_test))

# Reporte de clasificación.
print(classification_report(y_test, y_pred))

# Matriz de confusión.
print(confusion_matrix(y_test, y_pred))

# Decision Tree.
# Creando el modelo.
model_Tree = DecisionTreeClassifier()

# Alimentando los datos de entrenamiento.
model_Tree.fit(x_train, y_train)

# Prediciendo.
y_pred = model_Tree.predict(x_test)

# Calculando las accuracies
print("Accuracy de entrenamiento :", model_Tree.score(x_train, y_train))
print("Accuracy de testeo :", model_Tree.score(x_test, y_test))

# Reporte de clasificación.
print(classification_report(y_test, y_pred))

# Matriz de confusión.
print(confusion_matrix(y_test, y_pred))

# AdaBoost
# Creando el modelo.
model_Boost = AdaBoostClassifier()

# Alimentando los datos de entrenamiento.
model_Boost.fit(x_train, y_train)

# Prediciendo.
y_pred = model_Boost.predict(x_test)

# Calculando las accuracies
print("Accuracy de entrenamiento :", model_Boost.score(x_train, y_train))
print("Accuracy de testeo :", model_Boost.score(x_test, y_test))

# Reporte de clasificación.
print(classification_report(y_test, y_pred))

# Matriz de confusión.
print(confusion_matrix(y_test, y_pred))

