## 1: Modularizar la Inferencia de YOLOP

El script original de YOLOP (`detect.py` o similar) suele estar hecho para leer imágenes desde una carpeta o un archivo de video guardado. Necesitás transformarlo en una función o clase reutilizable.

* **Qué hacer:** Creá un script nuevo (ej. `yas_perception.py`) donde importes el modelo y cargues los pesos preentrenados `.pth`.
* **El objetivo:** Definir una función limpia que reciba un **NumPy Array** (el frame de la cámara en vivo) y te devuelva los tensores de salida (las máscaras y las bounding boxes).
* **Prueba de éxito:** Pasar un frame suelto usando OpenCV en un script aparte y que la función te devuelva las matrices procesadas sin escribir archivos en el disco.

---

## Paso 2: Conectar BeamNG y Extraer el primer Frame

Antes de meter control, necesitás que BeamNG sea tu "proveedor de video".

* **Qué hacer:** Instalá `beamngpy` y armá un script básico para spawnear un auto con una cámara frontal arriba del tablero.
* **Calibración Crítica:** Configurá la cámara de BeamNG con el mismo campo de visión (FOV) y resolución que usaste para validar YOLOP (o redimensionalo en Python). Anotá la distancia focal virtual de esa cámara (la vas a necesitar para calcular distancias).
* **Prueba de éxito:** Que corras el script de Python, se abra BeamNG, manejes el auto manualmente y veas en una ventana de OpenCV (`cv2.imshow`) lo mismo que ve el auto en tiempo real.

---

## Paso 3: Calibrar la Vista de Pájaro (IPM) y Distancias

Acá es donde transformás los píxeles en metros para que los algoritmos de control entiendan qué hacer.

* **Para el carril (LKA):** Poné el auto recto en una recta de BeamNG. Agarrá un frame de la máscara de carril de YOLOP y calculá la **matriz de homografía** (`cv2.getPerspectiveTransform`) seleccionando 4 puntos del carril para "estirarlos" y transformarlos en un rectángulo perfecto (vista de pájaro). Guardá esa matriz, va a ser fija.
* **Para el crucero (ACC):** Poné otro auto adelante a una distancia conocida (ej. 10 metros usando la telemetría del mapa) y fijate cuántos píxeles mide de ancho su bounding box en YOLOP. Usá eso para calibrar tu fórmula de distancia monocular:

$$d = \frac{f \cdot W_{real}}{w_{pixel}}$$



---

## Paso 4: Escribir la Lógica de Control (PIDs sueltos)

Creá los cerebros matemáticos que van a decidir cuánto acelerar o cuánto doblar. No los conectes al simulador todavía, armalos como funciones independientes.

* **Control Lateral (LKA):** Diseñá un controlador PID que reciba el error lateral $e_y$ (la distancia entre el centro del auto y el centro del carril calculado en tu vista de pájaro) y devuelva un valor de dirección entre `-1` (todo a la izquierda) y `1` (todo a la derecha).
* **Control Longitudinal (ACC):** Diseñá otro PID que reciba la distancia estimada al auto de adelante, la compare con la distancia segura deseada y devuelva valores para el acelerador (`throttle`) y el freno (`brake`).

---

## Paso 5: Cerrar el Bucle (Closed-Loop)

Es hora de juntar todo en el script principal sincrónico. La secuencia de cada iteración del bucle *while* tiene que ser estrictamente esta:

```
[BeamNG avanza 1 step] ➡️ [Captura Frame] ➡️ [Inferencia YOLOP] ➡️ [Cálculo de ey y d] ➡️ [PIDs calculan comandos] ➡️ [Enviar comandos a BeamNG]

```

* **Recomendación:** Empezá probando el LKA solo (que el auto siga el carril a velocidad constante). Cuando funcione perfecto y no zigzaguee, desactivá el LKA y probá el ACC en una línea recta atrás de otro auto. Cuando ambos funcionen por separado, activalos al mismo tiempo.

---

## Paso 6: El Núcleo del Paper (Análisis de Sensibilidad e Inputs Mínimos)

Una vez que el auto ya se maneja solo de forma estable combinando YOLOP + la telemetría de BeamNG (velocidad actual para ajustar los frenados, etc.), pasás a la fase de investigación pura para tu entrega de la facultad:

1. **Armar el Data Logger:** Hacé que tu script guarde en un CSV el timestamp, el error lateral, la oscilación del volante y si hubo colisiones o salidas de carril.
2. **Experimento 1 (Base):** Correr el sistema con todos los inputs (Cámara + Telemetría interna). Medir la precisión del control.
3. **Experimento 2 (Remover Telemetría - Hacia el Vision-Only):** Quitá el input de la velocidad del velocímetro del auto. Ahora obligá a tu código a *estimar* la velocidad del auto midiendo el flujo óptico de la ruta con la cámara o viendo cómo cambian de tamaño los objetos.
4. **Documentar el quiebre:** Seguir quitando variables hasta que el sistema empiece a fallar. El gráfico de cómo se degrada la performance a medida que sacás telemetría física para depender solo de la cámara va a ser el corazón de las conclusiones de tu paper.