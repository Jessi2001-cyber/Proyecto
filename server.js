// Importar las dependencias necesarias
const express = require('express');
const bodyParser = require('body-parser');
const firebase = require('firebase');

// Inicializar la aplicación de Express
const app = express();

// Configurar middleware
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// Configurar la conexión con Firebase
const firebaseConfig = {
  // Configuración de Firebase (apiKey, authDomain, etc.)
};

firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();

// Definir las rutas y controladores
app.get('/predictions', (req, res) => {
  // Obtener las predicciones desde la base de datos
  // Realizar las consultas necesarias en Firestore

  // Ejemplo de consulta de todas las predicciones en una colección
  db.collection('predictions')
    .get()
    .then((snapshot) => {
      const predictions = [];
      snapshot.forEach((doc) => {
        const prediction = doc.data();
        predictions.push(prediction);
      });
      res.json(predictions);
    })
    .catch((error) => {
      console.error('Error al obtener las predicciones:', error);
      res.status(500).json({ error: 'Error al obtener las predicciones' });
    });
});

app.post('/predictions', (req, res) => {
  // Guardar los datos ingresados en el formulario en la base de datos
  const { fruto, incidencia, severidad } = req.body;

  // Ejemplo de guardar los datos en Firestore
  db.collection('predictions')
    .add({
      fruto,
      incidencia,
      severidad,
    })
    .then(() => {
      res.json({ message: 'Datos guardados correctamente' });
    })
    .catch((error) => {
      console.error('Error al guardar los datos:', error);
      res.status(500).json({ error: 'Error al guardar los datos' });
    });
});

// Iniciar el servidor
const port = 3000;
app.listen(port, () => {
  console.log(`Servidor en funcionamiento en el puerto ${port}`);
});

