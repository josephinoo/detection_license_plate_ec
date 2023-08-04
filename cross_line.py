import cv2
import numpy as np
import datetime

# Cargar los archivos de configuración y pesos pre-entrenados de YOLO
net = cv2.dnn.readNet("yolov3-tiny.cfg", "yolov3-tiny.weights")

# Obtener las capas de salida de YOLO (detecciones)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Definir las clases de objetos que YOLO puede detectar
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Definir los colores de las cajas de detección
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Definir las coordenadas de la línea que el carro debe cruzar (un poco a la izquierda)
line_x = None
line_y = None

# Definir el desplazamiento hacia la izquierda (en píxeles)
left_offset = 120

# Crear una ventana de visualización
cv2.namedWindow("Object Detection")

# Capturar video desde la cámara
captura = cv2.VideoCapture("test3.mp4")

# Track car position to check for crossing the line
car_crossed_line = False

while True:
    # Leer el siguiente fotograma del video
    ret, frame = captura.read()

    if not ret:
        break

    # Obtener las dimensiones del fotograma
    height, width, _ = frame.shape

    # Dibujar la línea de cruce un poco a la izquierda
    if line_x is None:
        line_x = width // 2 - left_offset
        line_y = 0
    cv2.line(frame, (line_x, line_y), (line_x, height), (255, 0, 0), 2)

    # Pasar el fotograma por la red de YOLO para obtener las detecciones
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Procesar las detecciones obtenidas
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.75:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Filtrar las detecciones que están completamente dentro del fotograma
                if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                    # Check if the detected object is a car
                    if classes[class_id] == 'car':
                        # Check if the car is inside the line crossing ROI
                        if x <= line_x and x + w >= line_x:
                            # Draw bounding box and label for the detected car
                            color = colors[class_id]
                            label = f"{classes[class_id]}: {confidence:.2f}"
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # Check if the car crossed the line
                            if not car_crossed_line:
                                car_crossed_line = True
                                # Save the frame at the moment the car crossed the line
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_filename = f"car_capture_{timestamp}.jpg"
                                cv2.imwrite(save_filename, frame)

    # Reset the car_crossed_line variable at the end of each frame processing
    car_crossed_line = False

    # Mostrar el fotograma actual en la ventana
    cv2.imshow("Object Detection", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
captura.release()
cv2.destroyAllWindows()
