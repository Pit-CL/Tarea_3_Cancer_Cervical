# Tarea 3 Machine Learning - **Cervical Cancer Behavior Risk**
Magíster en Data Science - UDD
Alumno: Rafael Farías Poblete.
Profesor: Joaquín Villagra Pacheco.

# Contexto del dataset

Cada registro representa datos de seguimiento de casos positivos y negativos de cáncer cervico uterino. Estos pacientes han sido analizados por el Dr. Sobar del STIKES Indonesia Maju.

El data set tiene características mutivariables con atributos de datos enteros. El número de muestras es de 72 pacientes analizados en 18 atributos o variables independientes y una variable dependiente. El dataset no posee valores nulos. 

Como se dijo anteriormente el dataset tiene 18 atributos (proveniente de 8 variables, el nombre de la variable es la primera parte del nombre) 

1) behavior_eating 
2) behavior_personalHygine 
3) intention_aggregation 
4) intention_commitment 
5) attitude_consistency 
6) attitude_spontaneity 
7) norm_significantPerson 
8) norm_fulfillment 
9) perception_vulnerability 
10) perception_severity 
11) motivation_strength 
12) motivation_willingness 
13) socialSupport_emotionality 
14) socialSupport_appreciation 
15) socialSupport_instrumental 
16) empowerment_knowledge 
17) empowerment_abilities 
18) empowerment_desires 
19) ca_cervix (this is class attribute, 1=tiene cáncer cervical, 0=no tiene cáncer cervical.)

# Problema Objetivo.

El objetivo de este análisis es poder predecir a través de un **modelo de clasificación** si un paciente basado en los 18 atributos es propenso a tener cáncer (verdadero positivo). 

# Métricas de performance de los modelos.

Al ser un modelo de clasificación se utilizan las siguientes métricas de performance para poder determinar qué modelo es el que mejor se ajusta para darle respuesta al problema objetivo. Cada una de estas será abordada en el apartado de resultados.


1. Accuracy.
2. Confusion Matrix.
3. Precision.
4. Recall.
5. Roc curve.
# Análisis descriptivo de los datos.

Para poder describir los datos se utilizan una serie de funciones y gráficos descritos en el script entregado en anexos, entre ellos un .info() para poder obtener los tipos de variables y .describe() para obtener una descripción estadística simple del dataset, ver el comportamiento en la normalidad de los datos, entro otras. Sin embargo lo más importante de este dataset es poder identificar si está balanceado o no para poder aplicar un modelo de clasificación que sea más robusto frente a a la distribución de sus clases y así poder responder de mejor manera al objetivo.

En este caso de estudio y según la imagen 1 tenemos un dataset des balanceado. Por lo tanto se deberá tener especial cuidado en el modelo a utilizar ya que no todos los modelos de clasificación son capaces de trabajar de buena manera con datos no balanceados. En algunos modelos es necesario balancear estos datos de manera “manual” ; varios de los métodos pueden ser encontrados [aquí.](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/) 

![Imagen 1: Distribución de las clases objetivo (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607530078377_imagen.png)

# Estandarización y escalamiento de los futuros.

En el paso anterior de descripción de datos también se pudo notar que existen ciertos outliers, que pueden ser observados en la imagen 2. Por lo tanto es recomendable usar alguna función de estandarización con el fin de disminuir la sensibilidad del modelo a estos.

En el caso de este informe se utiliza StandarScaler de la librería Scikit-learn.  La librería indicada además permite escalar las variables con el fin de permitir llevarlas todas a un mismo rango de variabilidad, en este caso entre -1 y 1.


![Imagen 2: Gráfico de Boxplot para las variables estudiadas (Elaboración propia, 2020)](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607535907297_imagen.png)


Como método complementario buscaremos cuántos outliers hay en el dataset a través del método de DBSCAN, para ver si es importante eliminarlos o no. En primera instancia se busca el eps óptimo que permite determinar la frontera para determinar cuáles quedan fuera y poder ser catalogados como outliers a través de la metodología propuesta por [Nadia Rahmah at all, 2012](https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012).

En la siguiente imagen 3, se puede notar que el valor 5 de eps es el que más se acercaría a una solución óptima.


![Imagen 3: Eps Óptimo (Elaboración propia, 2020)](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607794023390_imagen.png)


Una vez aplicado el método de DSCAN (ver anexo) se obtiene un array en donde se pueden identificar sólo 2 outliers (entrada 15 y entrada 36) dentro del data set por lo que no serán eliminados.

# Seleccionando y entrenando el modelo.

Para hacerle frente de la mejor manera al desafío , este informe está basado en el estudio realizado por [Chao Chen at all (Department of Statistics, UC Berkeley, 2003)](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf), cuya librería es parte de [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.ensemble.BalancedRandomForestClassifier.html), en donde plantean que el mejor modelo a utilizar para clases des balanceadas puede ser el Weighted Random Forest o bien  Balanced Random Forest, de donde se cita parte de sus conclusiones “We presented two ways of learning imbalanced data based on random forest. **Weighted Random Forest** put more weights on the minority class, thus penalizing more heavily on misclassifying the minority class. **Balanced Random Forest** combines the down sampling majority class technique and the ensemble learning idea, artificially altering the class distribution so that classes are represented equally in each tree. From the experiments on various data sets, we can conclude that both Weighted RF and Balanced RF have performance superior to most of the existing techniques that we studied. Between WRF and BRF, however, there is no clear winner”. 

Por lo tanto para este informe se utiliza uno de los modelos el Balanced Random Forest al cuál se le agrega un trabajo en pipeline el que incluye StandarScaler,  Principal Component Analysis o PCA, el cuál le “entrega” al modelo BRF las variables que mejor explican el problema con el objetivo de balancear aún mejor la muestra al descartar variables que no aportan en un 40% y manteniendo aquellas que representan el 60% de la variabilidad en la información, esto  según lo explicado en el estudio realizado por [Nadir Mustafa at all (2017)](https://www.researchgate.net/publication/313799879_A_Classification_Model_for_Imbalanced_Medical_Data_based_on_PCA_and_Farther_Distance_based_Synthetic_Minority_Oversampling_Technique). 

De la siguiente imagen 4 detectamos el óptimo de PCA y en la imagen 5 se puede observar los ratios de la varianza explicada para los 3 componentes principales que representan un 60% de la información.

![Imagen 4: Número óptimo de PCA (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607632223128_imagen.png)



![Imagen 5: Ratios de varianza explicada para los componentes principales (Elaboración propia, 2020)](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607622571373_imagen.png)


El modelo anterior es comparado contra Support Vector Machine, Random Forest y Gradient Boost, en las mismas condiciones de pipeline, con el fin de demostrar que el modelo BRF es el que mejor se ajusta. Se escogen estos modelos ya que según recomendaciones son los más indicados de la librería sklearn para datasets des balanceados. 

Los datos son separados en datos de entrenamiento y test a través de la función train_split de sklearn con un tamaño de datos para testeo del 40%. Se separa en 40% debido a que con 20% y 30% se lograba una clasificación perfecta de BRF, por lo que se asume que al ser desbalanceado es necesario aumentar los datos de testeo, punto en cual el modelo comienza a generar errores, lo que lo acerca a un modelo más general. 

# Resultados.

Los resultados obtenidos de accuracy en el testeo son los indicados en la imagen 6, en donde se demuestra que el modelo SVM es el que obtiene el mejor accuracy de los modelos comparados.
Sin embargo al ser un data set desbalanceado, el accuracy no es el mejor indicador por sí solo.


![Imagen 6: Resultados de accuracy (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607622666203_imagen.png)


Por otro lado y como se comentó en el apartado sobre métricas de performance, es necesario comparar este indicador con otros para poder determinar el modelo que mejor predice la clase para el problema estudiado.

En la siguiente imagen 7 podemos observar las distintas matrices de confusión. 


![Imagen 7: Matrices de confusión para los diferentes clasificadores (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607622792166_imagen.png)


En este caso el modelo BRF es el mejor predictor, ya que al ser un caso de cáncer, en este estudio se cree que el modelo debe privilegiar que el clasificador sea capaz de identificar lo más cercano al 100% de los verdaderos positivos, es decir, de aquellos que tienen predicción de cáncer.

Para complementar la información, en la siguiente imagen 8 y 9 se muestran los resultados del reporte de clasificación del modelo BRF y SVM,  de esta manera se incluyen las métricas de performance de precision, recall y f1-score.  El modelo BRF es capaz de encontrar el 100% de los pacientes con cáncer y además predice el 82%  de los verdaderos positivos y verdaderos negativos, en este caso el trade-off son 2 falsos positivos. SVM en cambio comete errores en identificar aquellos verdaderos positivos, por lo que tiene un trade-off prediciendo un falso negativo, lo cuál para este caso de estudio es “fatal” al no predecir una persona con cáncer.


![Imagen 8: Reporte de clasificación para BRF (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607623053755_imagen.png)
![Imagen 9: Reporte de clasificación para SVM (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607623451306_imagen.png)


Por último comparamos las curvas ROC y el indicador AUC, ya que interesa preocuparse de la exactitud del modelo en determinar los Verdaderos Positivos, obteniendo los siguientes gráficos mostrados en la siguiente imagen 10 .  Del cálculo de AUC obtenemos un valor AUC para BRF de 0.95 (línea roja), seguido de SVM con 0,94 (línea azul), luego RF (línea naranja) con 0,88 y por último GradientBoost  con 0.75 (línea verde).  


![Imagen 10: Curva ROC (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607623700629_imagen.png)


Quizás de lo anterior se puede pensar que los datos tienen un overfitting al obtener resultados demasiado altos para BRF, esto se puede deber a que el dataset no contiene mucha información. 

Por lo tanto se vuelve a validar el modelo a través de Cross Validation, para revisar si éste se sigue ajustando y de qué manera, frente a un “stress” mayor en el split de los datos y se obtienen los siguientes resultados indicados en la imagen 11.


![Imagen 11: Cross Validation para SVM, RF, GBOOST y BRF (Elaboración propia, 2020).](https://paper-attachments.dropbox.com/s_463DB67259EA3D2783E77B81883E55F23279FE93B0829A502CB72A67CE556D28_1607623911238_imagen.png)


Se puede notar que la predicción del modelo BRF sigue siendo superior en cuanto a no fallar consistentemente en la detección de verdaderos positivos, seguido muy de cerca por el clasificador SVM.

# Conclusiones.

Luego de finalizado este informe se puede concluir que el Modelo Balanced Random Forest es el que mejor se ajusta al problema planteado (detectar el 100% de los que padecen cáncer) versus los otros modelos mostrados, obteniendo acurracy cercano al 93%, una matriz de confusión carente de falsos negativos (tienes cancer pero no lo predice), un f1-score bastante equilibrado entre precision y recall, un recall del 100% que indica la capacidad del modelo de predecir efectivamente aquellos con cáncer, un precision del 80%  y por último  un AUC cercano a uno.   Es importante notar el trade-off entre precision y recall el cual siempre dependerá del tipo de respuesta predictiva que se esté buscando. 

Además de lo anterior se puede demostrar que sólo a través de 4 componentes principales (Aplicación de PCA) se explica el 60% de la información, por lo que el computo es bastante rápido, además de traspasarle al modelo de clasificación datos más balanceados. 

Por otro lado se realiza un Cross Validation para simular un mayor “estress” en la predicción, obteniendo excelente resultados tanto con BRF como con SVM.

Por último y al no contar con una mayor cantidad de información, se recomienda que no se prefiera un modelo sobre otro, sino que seguir testeando ambos modelos a medida que se va obteniendo una mayor cantidad de datos. Sobre todo ya que según el estudio indicado de Chao Chen, recomienda lo siguiente “For any classifier, there is always a trade off between true positive rate and true negative rate; and the same applies for recall and precision. In the case of learning extremely imbalanced data, quite often the rare class is of great interest. In many applications such as drug discovery and **disease diagnosis**, it is desirable to have a classifier that gives high prediction accuracy over the minority class (Acc+), while maintaining reasonable accuracy for the majority class (Acc−). Weighted Accuracy is often used in such situations”, que en el caso de estudio los resultados obtenidos de Weighted Accuracy para el modelo BRF es de 0,93 y SVM 0,97.

# Anexos.
## Código.


    from sklearn.cluster import DBSCAN
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors
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
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from imblearn.metrics import classification_report_imbalanced
    import numpy as np
    
    # Archivo que contiene los datos
    file = '/home/rafaelfp/Dropbox/Postgrados/MDS/ML/ML2/Tarea_3/data/sobar-72.csv'
    
    # Se crea el df en pandas
    df = pd.read_csv(file)
    
    # Se revisa el data set
    print('Un resumen del DF\n', df.head())
    
    # Se revisa la info de la data para conocer las variables asociadas.
    print('Información de las variables\n', df.info())
    
    # Describe para ver rápidamente el comportamiento de las variables.
    print('Descripción estadística\n', df.describe())
    
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
    
    # Primer paso método DBSCAN para la detección de outliers.
    scaler = StandardScaler()
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(scaler.fit_transform(df))
    distances, indices = nbrs.kneighbors(scaler.fit_transform(df))
    
    # Se grafica para poder determinar el eps óptimo.
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.title('eps Óptimo a través de Nearest Neighbors')
    plt.xlabel('Distancia')
    plt.ylabel('eps')
    plt.show()
    
    # Se aplica el método DBSCAN con el eps encontrado.
    outlier_detection = DBSCAN(
        eps=5,
        metric="euclidean",
        min_samples=3,
        n_jobs=-1)
    clusters = outlier_detection.fit_predict(scaler.fit_transform(df))
    
    # Separando el df en x e y
    x = df.iloc[:, :18]  # Futures
    y = df.iloc[:, 19]  # Variable dependiente
    
    # Dividiendo el df en test y entrenamiento.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,
                                                        random_state=42)
    
    # Como en el pipeline traspasaré PCA al clasificador, se debe identificar los
    # PCA que mejor expliquen el problema.
    pca2 = PCA().fit(x)
    plt.plot(pca2.explained_variance_ratio_, linewidth=2)
    plt.xlabel('Componentes')
    plt.ylabel('Ratio de Varianza explicada')
    plt.title('Número de PCA óptimo')
    plt.show()
    
    # Creando los pipelines.
    pipe_svm = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=3)),
                         ('clf', svm.SVC(random_state=42))])
    
    pipe_RF = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=3)),
                        ('clf', RandomForestClassifier(random_state=42))])
    
    pipe_grad = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=3)),
                          ('clf', GradientBoostingClassifier(random_state=42))])
    
    pipe_BRF = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=3)),
                         ('clf', BalancedRandomForestClassifier(random_state=42))])
    
    # Lista de pipelines
    pipelines = [pipe_svm, pipe_RF, pipe_grad, pipe_BRF]
    
    # Fit a los pipelines
    for pipe in pipelines:
        pipe.fit(x_train, y_train)
    
    # Diccionario de los pipelines
    pipe_dict = {0: 'SVM', 1: 'RF', 2: 'GradientBoost', 3: 'BRF'}
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
    
    # Imprimiendo los reportes de clasificación.
    for idx, val in enumerate(pipelines):
        print('==================================================================')
        print('Reporte de clasificación para', pipe_dict[idx], 'es:\n',
              classification_report(y_test, val.predict(x_test)))
    
    print('======================================================================')
    
    # Se grafica ROC
    for idx, val in enumerate(pipelines):
        fpr, tpr, thresholds = roc_curve(y_test, val.predict(x_test))
        plt.plot(fpr, tpr, linewidth=2, label=pipe_dict[idx])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
    plt.show()
    
    # Se calcula AUC
    for idx, val in enumerate(pipelines):
        print('ROC AUC score para', pipe_dict[idx], 'es:\n',
              roc_auc_score(y_test, val.predict(x_test)))
    
    # Revisando con Cross Validation
    cv = KFold(5, shuffle=True, random_state=42)
    print('Realizando Cross Validation para los clasificadores')
    for pipe in pipelines:
        print('Resultado de CV',
              cross_validate(pipe, x, y,
                             cv=cv, scoring=('accuracy', 'f1',
                                             'roc_auc', 'recall')))
    
    # Otra manera de obtener un reporte de clasificación.
    # for idx, val in enumerate(pipelines):
    #     target_names = ['class 0', 'class 1']
    #     print('Reporte de clasificación para',
    #           pipe_dict[idx], 'es:\n',
    #           classification_report_imbalanced(y_test, val.predict(x_test),
    #                                            target_names=target_names))



