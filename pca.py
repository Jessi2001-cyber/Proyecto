import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    # Load the data from the pandas dataframe
    dt_heart = pd.read_csv('./data/base de datos.csv')
    
    # Print the first 5 records
    print(dt_heart.head(5))
    
    # Separate the features and the target variable
    dt_features = dt_heart.drop(['INCIDENCIA'], axis=1)
    dt_target = dt_heart['INCIDENCIA']
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    dt_features = imputer.fit_transform(dt_features)
    
    # Normalize the data
    scaler = StandardScaler()
    dt_features = scaler.fit_transform(dt_features)
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
    
    # Apply PCA
    pca = PCA(n_components=3)
    dt_train_pca = pca.fit_transform(X_train)
    dt_test_pca = pca.transform(X_test)
    
    # Apply logistic regression to the PCA-transformed data
    logistic_pca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_pca.fit(dt_train_pca, y_train)
    score_pca = logistic_pca.score(dt_test_pca, y_test)
    
    # Print the PCA results
    print("SCORE PCA: ", score_pca)
    
    # Apply Kernel PCA
    kernels = ['linear', 'poly', 'rbf']
    for kernel in kernels:
        kpca = KernelPCA(n_components=3, kernel=kernel)
        dt_train_kpca = kpca.fit_transform(X_train)
        dt_test_kpca = kpca.transform(X_test)
        
        # Apply logistic regression to the Kernel PCA-transformed data
        logistic_kpca = LogisticRegression(solver='lbfgs', max_iter=1000)
        logistic_kpca.fit(dt_train_kpca, y_train)
        score_kpca = logistic_kpca.score(dt_test_kpca, y_test)
        
        # Print the Kernel PCA results
        print("SCORE KPCA", kernel, ": ", score_kpca)
    
    # Apply Incremental PCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    dt_train_ipca = ipca.fit_transform(X_train)
    dt_test_ipca = ipca.transform(X_test)
    

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
        
        # Imprimir los resultados de Kernel PCA
        print("SCORE KPCA", kernel, ": ", score_kpca)
    
    # Aplicar Incremental PCA
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    dt_train_ipca = ipca.fit_transform(X_train)
    dt_test_ipca = ipca.transform(X_test)
    
    # Aplicar la regresión logística a los datos de Incremental PCA
    logistic_ipca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_ipca.fit(dt_train_ipca, y_train)
    score_ipca = logistic_ipca.score(dt_test_ipca, y_test)
    
    # Imprimir el resultado de Incremental PCA
    print("SCORE Incremental PCA: ", score_ipca)