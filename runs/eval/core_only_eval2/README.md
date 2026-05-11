# Resultados de evaluación (core_only_eval2)

Esta carpeta contiene los resultados finales solicitados de evaluación del modelo, restringidos a las **3 clases base** del proyecto:

- `Person`
- `Hardhat` (casco)
- `Safety Vest` (chaleco)

## Archivos principales

- Curva Precision–Recall (PR): [BoxPR_curve.png](BoxPR_curve.png)
	- Qué muestra: el trade-off entre **Precisión (P)** y **Recall (R)** para distintos umbrales.
	- Cómo leerla: cuanto más cerca esté del “esquina superior derecha”, mejor. En detección suele ser más informativa que ROC.

- Matriz de confusión: [confusion_matrix.png](confusion_matrix.png)
	- Qué muestra: en qué clases el modelo **acierta** y en cuáles se **confunde** (por ejemplo, confundir `Person` con otra clase).
	- Cómo leerla: la diagonal idealmente domina (muchos aciertos). Fuera de la diagonal = confusiones.

- Matriz de confusión normalizada: [confusion_matrix_normalized.png](confusion_matrix_normalized.png)
	- Qué muestra: lo mismo que la matriz de confusión, pero en **proporciones** (0–1), útil si las clases tienen diferente cantidad de ejemplos.
	- Cómo leerla: permite comparar clases aunque estén desbalanceadas.

- Curva ROC (presencia por imagen): [roc_curve_presence.png](roc_curve_presence.png)
	- Qué muestra: ROC calculada como **presencia/ausencia por imagen** (no por localización de bounding boxes).
	- Cómo leerla: AUC más alto = mejor capacidad de “detectar si la clase aparece en la imagen”.
	- Nota: en detección de objetos, la métrica estándar sigue siendo PR/mAP; esto se incluye solo si te lo exigen.

- AUC por clase (ROC presencia): [roc_auc_presence.csv](roc_auc_presence.csv)
	- Qué contiene: la tabla con el AUC numérico por clase (Hardhat/Person/Safety Vest) para citarlo en el informe.

## Notas rápidas

- Curvas por umbral:
	- [BoxP_curve.png](BoxP_curve.png): cómo cambia la **precisión** al variar el umbral de confianza.
	- [BoxR_curve.png](BoxR_curve.png): cómo cambia el **recall** al variar el umbral de confianza.
	- [BoxF1_curve.png](BoxF1_curve.png): equilibrio precisión/recall (F1) según el umbral; el pico sugiere un umbral “razonable”.

- Imágenes de ejemplo (sanity-check visual):
	- `val_batch*_pred.jpg`: ejemplos de predicciones del modelo en validación.
	- `val_batch*_labels.jpg`: etiquetas reales (ground truth) de los mismos lotes.

## Estado actual del modelo (qué se obtiene de estos resultados)

Con estos resultados puedes describir el **estado actual** de la app/modelo así:

- **Rendimiento global (detección + localización):** es **moderado**.
	- En validación, el modelo tiene un mAP50-95 aproximado de **0.340** (mientras que mAP50 es **0.666**), lo que sugiere que detecta “bastante” a IoU relajado, pero la precisión de localización baja al exigir IoU más estricto.

- **Casco (Hardhat):** es la clase más sólida.
	- Tiene buena precisión y recall (P≈0.831, R≈0.620) y el mejor mAP de las 3 clases.

- **Persona (Person):** funciona, pero es el punto más sensible para la lógica EPP.
	- Su recall es menor (R≈0.482). Si el modelo no detecta a la persona, no se puede evaluar correctamente si “tiene casco/chaleco” dentro de su bounding box.

- **Chaleco (Safety Vest):** desempeño aceptable, pero con recall moderado.
	- R≈0.463 implica que se pueden perder chalecos en algunas condiciones (oclusiones, iluminación, ángulos), lo cual puede generar falsos “FALTA CHALECO”.

- **ROC (presencia por imagen):** indica buena capacidad de “detectar si aparece la clase”, pero no sustituye a mAP.
	- Los AUC son altos (p.ej. Hardhat ≈ 0.927), lo que sugiere que como clasificador de presencia por imagen va bien; aun así, la métrica principal en detección sigue siendo PR/mAP.

**Conclusión operativa:** hoy el sistema es útil para monitoreo, pero es esperable que existan falsos “sin EPP” cuando:
- la persona no se detecta (recall de `Person`), o
- el chaleco/casco está parcialmente oculto o con baja visibilidad (recall de EPP).
