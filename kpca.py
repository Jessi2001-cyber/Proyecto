import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargar los datos del dataframe de pandas
    dt_heart = pd.read_csv('./data/base de datos.csv')
    
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
    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        kpca = KernelPCA(n_components=3, kernel=kernel)
        dt_train_kpca = kpca.fit_transform(X_train)
        dt_test_kpca = kpca.transform(X_test)
        
        # Aplicar la regresión logística a los datos de Kernel PCA
        logistic_kpca = LogisticRegression(solver='lbfgs', max_iter=1000)
        logistic_kpca.fit(dt_train_kpca, y_train)
        score_kpca = logistic_kpca.score(dt_test_kpca, y_test)
        
        # Imprimir los resultados
        print("SCORE KPCA", kernel, ": ", score_kpca)
    
    # Aplicar Incremental PCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    dt_train_ipca = ipca.fit_transform(X_train)
    dt_test_ipca = ipca.transform(X_test)
    
    # Aplicar la regresión logística a los datos de Incremental PCA
    logistic_ipca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_ipca.fit(dt_train_ipca, y_train)
    score_ipca = logistic_ipca.score(dt_test_ipca, y_test)
    
    # Imprimir los resultados
    print("SCORE IPCA: ", score_ipca)

