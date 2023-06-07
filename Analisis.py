import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargar los datos del dataframe de pandas
    dt_heart = pd.read_csv('./data/Datos_Agosto_Diciembre.csv')
    
    # Imprimir un encabezado con los primeros 5 registros
    print(dt_heart.head(5))
    
    # Guardar el dataset sin la columna de target
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    
    # Este será nuestro dataset, pero sin la columna
    dt_target = dt_heart['INCIDENCIA']
    
    # Normalizar los datos
    scaler = StandardScaler()
    dt_features = scaler.fit_transform(dt_features)
    
    # Partir el conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
    
    # Aplicar PCA
    pca = PCA(n_components=3)
    dt_train_pca = pca.fit_transform(X_train)
    dt_test_pca = pca.transform(X_test)
    
    # Aplicar Kernel PCA
    kernel_pca = KernelPCA(n_components=3, kernel='rbf')
    dt_train_kernel_pca = kernel_pca.fit_transform(X_train)
    dt_test_kernel_pca = kernel_pca.transform(X_test)
    
    # Aplicar Incremental PCA
    incremental_pca = IncrementalPCA(n_components=3, batch_size=10)
    dt_train_incremental_pca = incremental_pca.fit_transform(X_train)
    dt_test_incremental_pca = incremental_pca.transform(X_test)
    
    # Aplicar la regresión logística a los datos de PCA
    logistic_pca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_pca.fit(dt_train_pca, y_train)
    score_pca = logistic_pca.score(dt_test_pca, y_test)
    print("SCORE PCA: ", score_pca)
    
    # Aplicar la regresión logística a los datos de Kernel PCA
    logistic_kernel_pca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_kernel_pca.fit(dt_train_kernel_pca, y_train)
    score_kernel_pca = logistic_kernel_pca.score(dt_test_kernel_pca, y_test)
    print("SCORE Kernel PCA: ", score_kernel_pca)
    
    # Aplicar la regresión logística a los datos de Incremental PCA
    logistic_incremental_pca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_incremental_pca.fit(dt_train_incremental_pca, y_train)
    score_incremental_pca = logistic_incremental_pca.score(dt_test_incremental_pca, y_test)
    print("SCORE Incremental PCA: ", score_incremental_pca)

