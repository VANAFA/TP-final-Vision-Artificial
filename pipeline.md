# Pipeline de Integración: YOLOP + BeamNG.drive para ACC y LKA

Este documento detalla el pipeline de ingeniería para conectar tu modelo YOLOP ya replicado con el simulador **BeamNG.drive** utilizando **BeamNGpy**. El objetivo es cerrar el bucle de control (*closed-loop*) para ejecutar el Control de Crucero Adaptativo (ACC) y el Mantenimiento de Carril (LKA) con una única cámara frontal.

---

## Fase 1: Entorno de Simulación y Extracción de Datos (BeamNGpy)

El primer paso es levantar el simulador desde Python, configurar el vehículo y asegurar un flujo constante de imágenes y telemetría.

1. **Instalación de dependencias:** Asegurate de tener `beamngpy` y `opencv-python` instalados en tu entorno virtual de PyTorch.
2. **Configuración de BeamNG en Modo Sincrónico:** Esto es **crítico**. Si corre en modo asincrónico, el simulador avanzará mientras YOLOP procesa el frame (añadiendo latencia), haciendo que el auto choque. Usar `bng.set_deterministic()` y `bng.set_steps_per_second()`.
3. **Inicialización de Sensores:**
   * **Cámara Frontal:** Instanciar un sensor `Camera` detrás del parabrisas del vehículo (emulando la cámara del BYD). Configurar la resolución para que coincida o sea escalable a la entrada de YOLOP ($640 \times 384$).
   * **Sensor de Telemetría (Electrics):** Extraer variables como `wheelSpeed`, `throttle`, `brake`, `steering` y la aceleración longitudinal/lateral.

---

## Fase 2: Pipeline de Percepción (Inferencia de YOLOP)

Una vez que capturás el frame de BeamNG, pasa directo a tu red neuronal.

1. **Pre-procesamiento del Frame:**
   * Convertir el array de la cámara de BeamNG (RGBA/RGB) a formato BGR (si usás OpenCV) o directamente a Tensor de PyTorch.
   * Normalizar y redimensionar a las dimensiones exactas de tu modelo preentrenado.
2. **Inferencia en GPU:** Pasar el tensor por YOLOP para obtener las tres salidas en un solo *forward pass*:
   * **Detección:** Bounding boxes de los vehículos adelante.
   * **Área Conducible:** Máscara de segmentación del asfalto.
   * **Líneas de Carril:** Máscara de segmentación de las demarcaciones.

---

## Fase 3: Post-procesamiento y Extracción de Métricas de Control

Acá es donde transformás los píxeles de YOLOP en datos físicos reales (metros y radianes).

### 1. Mantenimiento de Carril (LKA)
* **IPM (Inverse Perspective Mapping):** Aplicar una matriz de homografía fija a la máscara de líneas de carril para obtener la **vista de pájaro (Bird's Eye View)**.
* **Ajuste Polinomial:** Detectar los píxeles de la línea izquierda y derecha, y ajustar un polinomio de 2do grado ($x = ay^2 + by + c$) para cada una.
* **Cálculo del Error Lateral ($e_y$):** Medir la distancia en píxeles (y luego pasarla a metros con la escala de la homografía) entre el centro del auto y el centro ideal del carril a una distancia de mirada fija (*look-ahead distance*).

### 2. Control de Crucero Adaptativo (ACC)
* **Filtrado de Bounding Box:** Identificar cuál de los autos detectados por YOLOP es el "vehículo objetivo" (el que está en nuestro mismo carril y más cerca).
* **Estimación de Distancia ($d$):** Usar el modelo de cámara estenopeica (*pinhole camera*) basándote en el ancho en píxeles de la bounding box del auto de adelante y el ancho real promedio de ese modelo en BeamNG:
  $$d = \frac{f \cdot W_{real}}{w_{pixel}}$$

---

## Fase 4: Lógica de Control (Closed-Loop Actuation)

Con el error lateral ($e_y$) y la distancia ($d$) calculados, se ejecutan los controladores que enviarán los comandos físicos de regreso al juego.

1. **Lógica LKA (Control Lateral):**
   * Pasar el error lateral $e_y$ por un controlador **PID** o un algoritmo **Pure Pursuit**.
   * El output será el ángulo de dirección del volante (`steering`) requerido para centrar el auto.
2. **Lógica ACC (Control Longitudinal):**
   * Definir una distancia segura de seguimiento basada en el tiempo (ej. $T_{gap} = 2 \text{ segundos} \implies d_{target} = v \cdot T_{gap}$).
   * Calcular el error de distancia ($e_d = d - d_{target}$).
   * Pasar $e_d$ por un PID longitudinal para determinar si se debe aplicar acelerador (`throttle`) o freno (`brake`).

---

## Fase 5: Bucle Principal en Python (Arquitectura del Script)

Estructuralmente, tu script principal de ejecución debería verse así en pseudocódigo:

```python
from beamngpy import BeamNGpy, Vehicle, sensors

# 1. Setup inicial
bng = BeamNGpy('localhost', 64215)
bng.open(launch=True)
vehicle = Vehicle('ego_vehicle', model='etk800')
camera = sensors.Camera('front_cam', bng, vehicle, ...) # Configurar posición

# 2. Loop de control sincrónico
try:
    while True:
        # Avanzar el simulador un paso de tiempo físico
        bng.step(steps=1) 
        
        # Capturar datos actuales
        sensors_data = bng.get_vehicle_sensors(vehicle)
        frame = sensors_data['front_cam']['colour']
        telemetry = vehicle.sensors['electrics']
        
        # Inferencia YOLOP
        lane_mask, target_bbox = run_yolop_inference(frame)
        
        # Post-procesamiento geométrico
        error_lateral = calculate_lane_error(lane_mask)
        distancia_auto = estimate_distance(target_bbox)
        
        # Controladores PID
        steering_cmd = pid_lateral(error_lateral)
        throttle_cmd, brake_cmd = pid_longitudinal(distancia_auto, telemetry['wheelSpeed'])
        
        # Enviar comandos de regreso al simulador
        vehicle.control(steering=steering_cmd, throttle=throttle_cmd, brake=brake_cmd)

finally:
    bng.close()