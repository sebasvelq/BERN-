# Clasificación de quejas de consumidores

## 1. Naive Bayes y TF-IDF

La base de datos de este trabajo se extrajo de la página data.gov, específicamente de La Oficina para la Protección Financiera del Consumidor de los Estados Unidos (CFPB). Las bases de datos contienen información de los consumidores y de las quejas que fueron reportadas al CFPB para luego ser reenviadas a las empresas. La base de datos contenía categorías; en este estudio se usaron las siguientes: “Fecha recibida”, “asunto”, “narrativa de la queja del consumidor”.

Para clasificar satisfactoriamente los reclamos de los consumidores se aplicaron varias técnicas de procesamiento de texto: conversiones a minúscula, eliminación de puntuación, lematización. Estas técnicas fueron aplicadas con el objetivo de simplificar el análisis de los textos.

Se realizó un método de vectorización TF-IDF mediante la biblioteca TfidVectorizer de Python para transformar la matriz TF-IDF a un máximo de 5000 características. La razón por la cual se eligió TF-IDF es que es ideal para Naive Bayes, ya que el modelo considera independencia entre características. Los datos fueron divididos entre datos de prueba y entrenamiento, el conjunto de prueba es del 20%.

La gráfica de distribución de texto muestra que el modelo está entrenado en su gran mayoría por textos cortos. Las gráficas muestran que las clases no están correctamente distribuidas, por lo que es recomendable balancear el dataset. Las gráficas de prueba y entrenamiento (verde y rojo, respectivamente) muestran una distribución similar, lo que es adecuado. La matriz de confusión y la distribución de errores muestran que la categoría 1 y la categoría 2 tienen una alta frecuencia de errores, por lo que se recomienda aplicar técnicas de ponderación para que el modelo entrene más en estas dos categorías.

El entrenamiento de Naive Bayes arrojó buenos resultados, sobre todo para la categoría de "Credit reporting, credit repair services." La precisión macro promedio fue del 68% y el F1 Score fue del 60%, pero este modelo flaquea con categorías con menos datos en sus respectivas clasificaciones.

Para mejorar la predicción de este modelo se pueden realizar validaciones cruzadas en los datos de entrenamiento y prueba, probar reducciones dimensionales como método de vectorización, como PCA o LSA, ya que estos métodos reducen la complejidad de la matriz TF-IDF, lo que permite mejorar la clasificación. También se pueden analizar más detenidamente las categorías con más errores de la matriz de confusión. Además, se puede eliminar el conjunto de palabras que no son frecuentes; para ello, se puede utilizar análisis de n-gramas para detectar patrones comunes de expresiones en los consumidores.

## 2. BERT

Un método de clasificación más potente es el método BERT (Bidirectional Encoder Representations from Transformers). Es un modelo de aprendizaje profundo que consiste en analizar las líneas de texto de forma bidireccional, mejorando la comprensión de palabras ambiguas. En este trabajo, el modelo BERT desarrollado clasificó los textos en categorías específicas. Primero se convierte la categoría de textos en números, luego se usa BertTokenizer para tokenizar los datos; posteriormente, se carga el modelo BERT preentrenado por Google y finalmente se definen métricas de evaluación como F1 y accuracy.

Los datos de pérdida de entrenamiento y validación muestran que la pérdida de entrenamiento disminuye consistentemente; lo mismo sucede para la pérdida de validación. Esto sugiere que el modelo está aprendiendo bien a medida que se entrena, aunque en el caso de la pérdida de validación la disminución es leve en comparación con la pérdida de entrenamiento, lo cual puede indicar un ligero sobreajuste.

En la gráfica de pérdida de entrenamiento y validación vemos que la pérdida de validación no baja tan drásticamente como la de entrenamiento, lo cual indica un posible sobreajuste. Para mejorar este inconveniente se recomienda ajustar hiperparámetros o implementar técnicas de dropout.

El modelo mostró una precisión general del 70% y un F1 Score del 70%, mejorando las predicciones de Naive Bayes. Sin embargo, se siguió obteniendo un menor porcentaje de predicción para las categorías con menos datos para analizar.

Como posibles mejoras para el modelo BERT se pueden probar algunos modelos derivados, también se recomienda aumentar la cantidad de datos que se están usando en este análisis y analizar las categorías con más errores para realizar un análisis particular a esas categorías. Lamentablemente, realizar estas mejoras conlleva un mayor costo computacional que conllevan estas modificaciones.
