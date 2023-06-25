import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/base de datos.csv')
    
    # Verificar y eliminar filas con valores no numéricos en la columna 'INCIDENCIA'
    dt_heart = dt_heart[pd.to_numeric(dt_heart['INCIDENCIA'], errors='coerce').notnull()]
    
    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']
    
    # Convertir valores categóricos en la columna 'INCIDENCIA' a numéricos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Imputar los valores faltantes en X
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(x)
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(x_imputed, y_encoded, test_size=0.35, random_state=1)

    estimators = range(2, 300, 2)
    total_mse = []
    best_result = {'mse': float('inf'), 'n_estimator': 1}

    for i in estimators:
        boost = GradientBoostingRegressor(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)
        new_mse = mean_squared_error(boost_pred, y_test)
        total_mse.append(new_mse)
        
        if new_mse < best_result['mse']: 
            best_result['mse'] = new_mse
            best_result['n_estimator'] = i
    
    print(best_result)


