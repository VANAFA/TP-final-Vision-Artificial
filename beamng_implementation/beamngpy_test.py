import os
import cv2
import numpy as np
import time
import mss
import keyboard
import csv
import math
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics

def main():
    # 1. Configurar directorios del dataset
    base_dir = "./beamng_dataset"
    img_dir = os.path.join(base_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    # Archivo CSV para guardar los metadatos sincronizados
    csv_file_path = os.path.join(base_dir, "dataset_log.csv")
    
    # Escribir el encabezado del CSV si el archivo no existe o está vacío
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['frame_id', 'speed_ms', 'steering', 'throttle', 'brake'])

    # 2. Inicializar conexión con BeamNG
    bng = BeamNGpy(
        'localhost', 
        64215,
        home=r'C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive',
        user=r'C:\Users\nalli\AppData\Local\BeamNG.drive_api'
    )
    
    print("[Sistema] Conectando a BeamNG...")
    bng.open(launch=True)

    # 3. Crear escenario
    scenario = Scenario('grille_autobahn_loop', 'yolop_data_generation')
    vehicle = Vehicle('ego_vehicle', model='etk800', license='DATA-GEN')

    # Configurar y adjuntar el sensor 'Electrics' para extraer la telemetría (pedales, volante, velocidad)
    electrics = Electrics()
    vehicle.sensors.attach('electrics', electrics)

    # Coordenadas de spawn exactas en la Autobahn
    spawn_pos = (-798.454, -4238.545, 26.060) 
    
    # Conversión del ángulo Euler (Z = 123.136 grados) a Cuaternión (x, y, z, w)
    yaw_rad = math.radians(123.136)
    qz = math.sin(yaw_rad / 2)
    qw = math.cos(yaw_rad / 2)

    scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=(0.0, 0.0, qz, qw), cling=True)

    # Construir escenario
    scenario.make(bng)

    print("[Simulador] Cargando el escenario...")
    bng.scenario.load(scenario)
    bng.scenario.start()

    # --- ACTIVAR TRÁFICO DENSO ---
    print("[Generador] Generando MUCHO tráfico (esto puede tardar unos segundos)...")
    try:
        bng.traffic.spawn(max_amount=20) # Subimos el límite a 20 para autopistas
    except Exception as e:
        print(f"[Aviso] No se pudo generar tráfico: {e}.")

    # Configurar el auto Ego (para que lo manejes vos o la IA)
    vehicle.set_shift_mode("realistic_automatic")
    # Si querés manejar VOS manualmente para enseñarle al auto (Behavioral Cloning), 
    # comentá estas 3 líneas de abajo. Si querés que maneje la IA, dejalas.
    vehicle.ai.set_mode('span')
    vehicle.ai.set_speed(35) # 126 km/h
    vehicle.ai.set_aggression(0.8)

    print("\n========================================================")
    print("🔧 CONTROLES Y HOTKEYS 🔧")
    print("1. Poné el juego en Pantalla Completa (Alt + Enter).")
    print("2. Ocultá la interfaz con 'Alt+U'.")
    print("3. Poné la cámara en Primera Persona (C).")
    print("--------------------------------------------------------")
    print("🟢 Presioná 'R' para COMENZAR a grabar (Start)")
    print("🟡 Presioná 'T' para PAUSAR la grabación (Pause)")
    print("🔴 Presioná 'Q' para SALIR y guardar todo (Quit)")
    print("========================================================\n")

    frame_count = 0
    recording = False

    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1] 
            
            # Abrimos el CSV en modo 'append' para ir guardando línea por línea
            with open(csv_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                while True:
                    # Chequeo de Hotkeys
                    if keyboard.is_pressed('r') and not recording:
                        recording = True
                        print("[REC] Grabación INICIADA. Recopilando datos...")
                        time.sleep(0.3) # Evitar rebote de la tecla
                    
                    elif keyboard.is_pressed('t') and recording:
                        recording = False
                        print("[PAUSE] Grabación PAUSADA.")
                        time.sleep(0.3)
                        
                    elif keyboard.is_pressed('q'):
                        print("[EXIT] Comando de salida recibido. Terminando...")
                        break

                    # Si estamos grabando, ejecutamos el pipeline
                    if recording:
                        # 1. Capturar Pantalla
                        sct_img = sct.grab(monitor)
                        img_bgra = np.array(sct_img)
                        img_bgr = img_bgra[:, :, :3]
                        img_resized = cv2.resize(img_bgr, (640, 480))
                        
                        # 2. Consultar Sensores (Telemetría actual)
                        vehicle.sensors.poll('electrics')
                        telemetry = vehicle.sensors.data['electrics']
                        
                        speed_ms = telemetry.get('wheelspeed', 0.0)
                        steering = telemetry.get('steering', 0.0)
                        throttle = telemetry.get('throttle', 0.0)
                        brake = telemetry.get('brake', 0.0)
                        
                        # 3. Guardar Imagen
                        frame_name = f"frame_{frame_count:06d}.jpg" # Formato a 6 dígitos para datasets grandes
                        cv2.imwrite(os.path.join(img_dir, frame_name), img_resized)
                        
                        # 4. Guardar Fila en CSV
                        csv_writer.writerow([frame_name, round(speed_ms, 3), round(steering, 4), round(throttle, 3), round(brake, 3)])
                        
                        if frame_count % 50 == 0:
                            print(f" -> Guardados {frame_count} frames... (Vel: {speed_ms*3.6:.1f} km/h | Volante: {steering:.2f})")

                        frame_count += 1
                        
                        # Tasa de captura: ~10 FPS a 20 FPS (Suficiente para Behavioral Cloning)
                        time.sleep(0.05)
                    else:
                        # Si no estamos grabando, dormir el hilo un poco para no saturar la CPU
                        time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Generador] Interrupción forzada.")
    except Exception as e:
        print(f"\n[ERROR] Crash en el bucle: {e}")

    finally:
        print("[Generador] Cerrando simulación...")
        bng.close()
        print(f"[Generador] Proceso terminado. Dataset y Metadatos guardados en: {base_dir}")

if __name__ == "__main__":
    main()