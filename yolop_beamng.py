import os
import cv2
import torch
import onnxruntime as ort
import numpy as np
from lib.core.general import non_max_suppression

# Asumimos que beamngpy está instalado y configurado en tu entorno
from beamngpy import BeamNGpy, Vehicle
from beamngpy.sensors import Camera

class YOLOPDetector:
    def __init__(self, weight_path="./weights/yolop-640-640.onnx"):
        # Inicializar ONNX una sola vez
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(weight_path)
        print(f"[YOLOP] Modelo cargado con éxito desde {weight_path}")
        
    def resize_unscale(self, img, new_shape=(640, 640), color=114):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        canvas = np.zeros((new_shape[0], new_shape[1], 3))
        canvas.fill(color)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        new_unpad_w, new_unpad_h = new_unpad[0], new_unpad[1]
        pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h

        dw = pad_w // 2
        dh = pad_h // 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

        canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img
        return canvas, r, dw, dh, new_unpad_w, new_unpad_h

    def process_frame(self, img_bgr):
        """
        Recibe un frame directo de la memoria (NumPy Array BGR)
        Devuelve: boxes (detecciones), da_seg_mask (área), ll_seg_mask (carril)
        """
        height, width, _ = img_bgr.shape

        # Pasar a RGB (como pide YOLOP)
        img_rgb = img_bgr[:, :, ::-1].copy()

        # Tu lógica de resize y normalización exacta
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = self.resize_unscale(img_rgb, (640, 640))

        img = canvas.copy().astype(np.float32)
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)

        # Inferencia en memoria
        det_out, da_seg_out, ll_seg_out = self.ort_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": img}
        )

        # Post-procesamiento de Bounding Boxes con PyTorch NMS
        det_out = torch.from_numpy(det_out).float()
        boxes = non_max_suppression(det_out)[0]
        boxes = boxes.cpu().numpy().astype(np.float32)

        # Si no hay cajas, dejamos un array vacío con la estructura correcta para que no crashee el loop
        if boxes.shape[0] > 0:
            boxes[:, 0] -= dw
            boxes[:, 1] -= dh
            boxes[:, 2] -= dw
            boxes[:, 3] -= dh
            boxes[:, :4] /= r

        # Post-procesamiento de Máscaras de Segmentación
        da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        da_seg_mask = np.argmax(da_seg_out, axis=1)[0]
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]

        # Escalar las máscaras binarias al tamaño original de la pantalla de BeamNG
        da_seg_mask = (da_seg_mask * 255).astype(np.uint8)
        da_seg_mask = cv2.resize(da_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
        ll_seg_mask = cv2.resize(ll_seg_mask, (width, height), interpolation=cv2.INTER_LINEAR)

        return boxes, da_seg_mask, ll_seg_mask


# =====================================================================
# BUCLE PRINCIPAL DE INTEGRACIÓN CON BEAMNG
# =====================================================================
def main():
    # 1. Inicializar el detector YOLOP
    detector = YOLOPDetector(weight_path="./weights/yolop-640-640.onnx")

    # 2. Setup de BeamNG y el vehículo
    # Asegurate de que los paths a tu instalación de BeamNG sean los correctos
    bng = BeamNGpy('localhost', 64215)
    bng.open(launch=True)

    # Spawnear vehículo (ej: un ETK800 o el que uses de pruebas)
    vehicle = Vehicle('ego_vehicle', model='etk800', licenPlate='VISION-UdeSA')
    
    # Configurar el escenario
    # Reemplazar 'west_coast_usa' por el mapa donde estés haciendo los tests
    bng.load_scenario('west_coast_usa') 
    bng.start_scenario()
    
    # 3. Setup de la cámara frontal (Emulando el ADAS monocromático del BYD)
    # Ubicación física aproximada en el parabrisas/retrovisor
    pos = (0, 1.7, 1.2)  
    direction = (0, 1, 0) # Mirando al frente
    up = (0, 0, 1)
    
    # Pedir resolución nativa fácil de procesar o escalar
    front_camera = Camera('front_cam', bng, vehicle, 
                          requested_capabilities=['colour'],
                          pos=pos, dir=direction, up=up,
                          fov=70, resolution=(640, 480))

    print("[Simulador] Entorno conectado. Iniciando bucle sincrónico...")
    
    # Forzar modo determinista para que la física espere a la IA
    bng.set_deterministic() 
    bng.set_steps_per_second(60)

    try:
        while True:
            # Avanzar la física del juego un paso técnico
            bng.step(1)
            
            # 4. Captura de datos de los sensores
            sensors_data = bng.get_vehicle_sensors(vehicle)
            
            # BeamNG suele devolver imágenes en formato RGBA o RGB
            frame_raw = sensors_data['front_cam']['colour']
            img_bgr = cv2.cvtColor(np.array(frame_raw), cv2.COLOR_RGB2BGR)
            
            # Obtener datos de la telemetría interna del auto
            electrics = vehicle.sensors['electrics']
            speed = electrics['wheelSpeed'] # Velocidad en m/s
            steering = electrics['steering']
            
            # 5. Inferencia de YOLOP en memoria (Pasamos el frame crudo)
            boxes, da_mask, ll_mask = detector.process_frame(img_bgr)

            # 6. Playground para tus algoritmos de control (LKA & ACC)
            # Acá abajo es donde procesás las variables que devuelve YOLOP:
            # - Con 'll_mask' calculás tu error lateral (ey) mediante IPM.
            # - Con 'boxes' buscás el auto más cercano y calculás su distancia (d).
            
            print(f"[Telemetría] Vel: {speed:.2f} m/s | Bounding Boxes detectadas: {len(boxes)}")

            # Visualización rápida en tiempo real para debuggear sin escribir a disco
            # Creamos un merge rápido en pantalla
            debug_frame = img_bgr.copy()
            debug_frame[da_mask > 0] = debug_frame[da_mask > 0] * 0.7 + np.array([0, 255, 0], dtype=np.uint8) * 0.3
            debug_frame[ll_mask > 0] = debug_frame[ll_mask > 0] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.astype(int)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
            cv2.imshow("ADAS Dashboard - UdeSA", debug_frame)
            
            # Romper el loop con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cv2.destroyAllWindows()
        bng.close()
        print("[Simulador] Conexión cerrada limpiamente.")

if __name__ == "__main__":
    main()