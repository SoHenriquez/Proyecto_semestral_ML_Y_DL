# Proyecto_semestral_ML_Y_DL
Este repositorio contiene el código para entrenar y evaluar tres modelos de aprendizaje automático para predecir la suscripción de un cliente a un servicio.

## Los modelos son:

- Random Forest
- SVC
- Redes neuronales

---

- El código se encuentra en el archivo main.py. 
- El archivo data.csv contiene los datos de entrenamiento y prueba. 
- El archivo dockerfile se utiliza para crear una imagen de Docker que contiene el código y los datos.
---
## Cómo usar el repositorio

### Para entrenar los modelos, siga estos pasos:

- Clone el repositorio en su computadora local.
- Instale las dependencias necesarias:
  ```
  pip install -r requirements.txt
  ```
- Ejecute el siguiente comando para entrenar los modelos:
  ```
  python main.py
  ```
Los resultados del entrenamiento se imprimirán en la consola.

### Para ejecutar los modelos, siga estos pasos:

- Construya la imagen de Docker:
```docker build -t my-model .```
- Ejecute el siguiente comando para ejecutar los modelos:
```docker run --rm my-model```

Los resultados de la predicción se imprimirán en la consola.

---
### Próximos pasos

En el futuro, se planea agregar los siguientes elementos al repositorio:

- Una interfaz de usuario para visualizar los resultados
- La capacidad de entrenar los modelos en un conjunto de datos más grande
- La capacidad de utilizar los modelos en un entorno de producción
## Contacto
```esteban.soto1901@alumnos.ubiobio.cl```
