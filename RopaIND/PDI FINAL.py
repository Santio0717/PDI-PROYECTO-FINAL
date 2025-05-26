import os 
import random
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk

carpeta_imagenes = "PDI FINAL"

def cargar_imagen_con_alfa(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")

    if len(imagen.shape) == 2:
        raise ValueError(f"La imagen {ruta} es en escala de grises, se esperan imágenes en color.")

    canales = imagen.shape[2]

    if canales == 4:
        return imagen
    elif canales == 3:
        b, g, r = cv2.split(imagen)
        alfa = np.ones(b.shape, dtype=b.dtype) * 255
        imagen_con_alfa = cv2.merge((b, g, r, alfa))
        return imagen_con_alfa
    else:
        raise ValueError(f"Formato no soportado para la imagen: {ruta}, canales: {canales}")

def elegir_imagen_aleatoria(prefijo):
    archivos = [f for f in os.listdir(carpeta_imagenes)
                if f.startswith(prefijo) and (f.endswith('.png') or f.endswith('.PNG'))]
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos para {prefijo} en {carpeta_imagenes}")
    elegido = random.choice(archivos)
    ruta_completa = os.path.join(carpeta_imagenes, elegido)
    return cargar_imagen_con_alfa(ruta_completa)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

pantalon_img = elegir_imagen_aleatoria("Pantalon")
camisa_img = elegir_imagen_aleatoria("Camisa")
sombrero_img = elegir_imagen_aleatoria("Sombrero")
instrumento_img = elegir_imagen_aleatoria("Flauta")

def insertar_superposicion(base, elemento, ubicacion, escala):
    try:
        i, j = ubicacion
        alto, ancho = escala

        if i is None or j is None or alto is None or ancho is None:
            return

        i = int(i)
        j = int(j)
        alto = int(alto)
        ancho = int(ancho)

        if ancho <= 0 or alto <= 0:
            return

        # Ajuste para que no se salga del borde
        if i < 0:
            ancho += i
            i = 0
        if j < 0:
            alto += j
            j = 0

        if ancho <= 0 or alto <= 0:
            return

        max_alto, max_ancho = base.shape[:2]
        if i + ancho > max_ancho:
            ancho = max_ancho - i
        if j + alto > max_alto:
            alto = max_alto - j

        if ancho <= 0 or alto <= 0:
            return

        # Redimensionar el elemento a superponer
        elemento = cv2.resize(elemento, (ancho, alto), interpolation=cv2.INTER_AREA)

        # Si no tiene canal alfa, lo ponemos opaco
        if elemento.shape[2] < 4:
            base[j:j+alto, i:i+ancho] = elemento
            return

        # Separa canales incluyendo alfa
        b, g, r, a = cv2.split(elemento)
        # Crear máscaras binarias para alfa
        alfa_normalizado = a / 255.0
        alfa_inv = 1.0 - alfa_normalizado

        # Extraemos la región de interés (ROI) de la base
        roi = base[j:j+alto, i:i+ancho].astype(float)

        # Convertir canales a float para mezcla
        b = b.astype(float)
        g = g.astype(float)
        r = r.astype(float)

        # Composición alfa para cada canal
        for c, canal in enumerate([b, g, r]):
            roi[..., c] = (alfa_normalizado * canal) + (alfa_inv * roi[..., c])

        # Convertir roi de float a uint8
        base[j:j+alto, i:i+ancho] = roi.astype(np.uint8)

    except Exception as e:
        print(f"Error en insertar_superposicion: {e}")

def mostrar_popup_personalizado():
    root = tk.Tk()
    root.title("Mensaje")
    root.configure(bg="#D2B48C")  # Café claro
    root.geometry("400x180")
    root.resizable(False, False)

    # Centrar ventana
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")

    mensaje_es = "GRACIAS POR APRENDER DE LAS CULTURAS INDÍGENAS 🪶, SI VUELVES A USAR LA EXPERIENCIA VERÁS COMO LA ROPA CAMBIA"
    mensaje_wayuu = "Aú, wayuu, jarapüi pa'ülsia vicha jachinü, ke te'eli e'ejomü vichan no'oni, juluu rinjuu."

    label_es = tk.Label(root, text=mensaje_es, bg="#D2B48C", fg="black", font=("Arial", 14, "bold"),
                        wraplength=380, justify="center")
    label_es.pack(padx=20, pady=(20, 5))

    label_wayuu = tk.Label(root, text=mensaje_wayuu, bg="#D2B48C", fg="black", font=("Arial", 13),
                          wraplength=380, justify="center")
    label_wayuu.pack(padx=20, pady=(0, 20))

    boton = tk.Button(root, text="Cerrar", bg="#8B5E3C", fg="white", font=("Arial", 12, "bold"),
                      command=root.destroy, padx=10, pady=5)
    boton.pack()

    root.mainloop()

def mostrar_mensaje_salida():
    mostrar_popup_personalizado()

camara = cv2.VideoCapture(0)

while True:
    exito, frame = camara.read()
    if not exito:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            palma_x = int(hand_landmarks.landmark[9].x * w)
            palma_y = int(hand_landmarks.landmark[9].y * h)
            insertar_superposicion(frame, instrumento_img, (palma_x - 100, palma_y - 60), (180, 180))

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_model.detectMultiScale(gris, 1.1, 4)
    for (a, b, c, d) in rostros:
        ancho_pant = int(c * 3.5)
        alto_pant = int(d * 4)
        pos_x_pant = a - int((ancho_pant - c) / 1.6)
        pos_y_pant = b - int(alto_pant * -1.1)
        insertar_superposicion(frame, pantalon_img, (pos_x_pant, pos_y_pant), (ancho_pant, alto_pant))

        ancho_cam = int(c * 4)
        alto_cam = int(d * 3.5)
        pos_x_cam = a - int((ancho_cam - c) / 2.4)
        pos_y_cam = b + d - int(d * 0.12)
        insertar_superposicion(frame, camisa_img, (pos_x_cam, pos_y_cam), (ancho_cam, alto_cam))

        ancho_sombrero = int(c * 1.2)
        alto_sombrero = int(d * 1.2)
        pos_x_sombrero = a - int((ancho_sombrero - c) / 2)
        pos_y_sombrero = b - int(alto_sombrero * 0.85)
        insertar_superposicion(frame, sombrero_img, (pos_x_sombrero, pos_y_sombrero), (ancho_sombrero, alto_sombrero))

    cv2.imshow("Detección de Manos y Ropa", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('x') or tecla == ord('X'):
        mostrar_mensaje_salida()  # Mostrar popup personalizado antes de salir
        break

    if cv2.getWindowProperty("Detección de Manos y Ropa", cv2.WND_PROP_VISIBLE) < 1:
        mostrar_mensaje_salida()  # Mostrar popup personalizado si cierran ventana
        break

camara.release()
cv2.destroyAllWindows()

#Para correrlo debo usar en terminal python "PDI FINAL.py" 

