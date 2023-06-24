import cv2
import numpy as np

# Cargar los archivos de configuración y pesos pre-entrenados de YOLO
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Obtener las capas de salida de YOLO (detecciones)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Definir las clases de objetos que YOLO puede detectar
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definir los colores de las cajas de detección
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Definir las coordenadas del cuadro de interés (ROI)
x_roi, y_roi, width_roi, height_roi = 100, 100, 300, 300

# Crear una ventana de visualización
cv2.namedWindow("Object Detection")

# Capturar video desde la cámara
captura = cv2.VideoCapture(0)

while True:
    # Leer el siguiente fotograma del video
    ret, frame = captura.read()

    # Dibujar el cuadro de interés (ROI)
    cv2.rectangle(frame, (x_roi, y_roi), (x_roi + width_roi, y_roi + height_roi), (0, 255, 0), 2)

    # Obtener la región de interés (ROI)
    roi = frame[y_roi:y_roi + height_roi, x_roi:x_roi + width_roi]

    # Redimensionar el ROI para el procesamiento más rápido
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pasar el ROI por la red de YOLO para obtener las detecciones
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar las detecciones obtenidas
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width_roi)
                center_y = int(detection[1] * height_roi)
                w = int(detection[2] * width_roi)
                h = int(detection[3] * height_roi)

                x = int(center_x - w / 2) + x_roi
                y = int(center_y - h / 2) + y_roi

                # Filtrar las detecciones que están completamente dentro del cuadro
                if x > x_roi and y > y_roi and x + w < x_roi + width_roi and y + h < y_roi + height_roi:
                    # Agregar la detección a las listas
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

                    # Dibujar la caja de detección y etiqueta correspondiente
                    color = colors[class_id]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, classes[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el fotograma actual en la ventana
    cv2.imshow("Object Detection", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
captura.release()
cv2.destroyAllWindows()
