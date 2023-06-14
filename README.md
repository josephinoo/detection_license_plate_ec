# Detección de Matrículas de Vehículos en Ecuador


Este proyecto proporciona un sistema para detectar y reconocer las matrículas de vehículos en Ecuador. El código está basado en el modelo YOLOv8 de Ultralytics.

## Pasos para ejecutar el código

Asegúrate de tener instalado Python, Git y virtualenv en tu sistema operativo. A continuación, sigue estos pasos:

1. **Clonar el repositorio**

    Abre la terminal e introduce el siguiente comando para clonar el repositorio:

    ```
    git clone https://github.com/josephinoo/detection_license_plate_ec.git
    ```

2. **Navegar hasta la carpeta clonada**

    Utiliza este comando para cambiar a la carpeta del proyecto:

    ```
    cd detection_license_plate_ec
    ```

3. **Crear un entorno virtual**

    Recomendamos usar un entorno virtual para evitar conflictos entre las bibliotecas necesarias para este proyecto y las que puedas tener instaladas globalmente en tu sistema. Para crear un entorno virtual con virtualenv, ejecuta el siguiente comando:

    ```
    python -m virtualenv venv
    ```

    Luego, actívalo con uno de los siguientes comandos, según tu sistema operativo:

    - En Windows:

        ```
        .\venv\Scripts\activate
        ```

    - En Unix o MacOS:

        ```
        source venv/bin/activate
        ```

4. **Instalar las dependencias**

    Este proyecto requiere algunas bibliotecas de Python. Puedes instalarlas con el siguiente comando:

    ```
    pip install -e '.[dev]'
    ```

5. **Configurar el directorio**

    Antes de iniciar la detección, es necesario ir al directorio donde se encuentra el archivo `predict.py`:

    ```
    cd ultralytics/yolo/v8/detect
    ```

6. **Ejecutar la detección**

    Finalmente, puedes iniciar la detección con el siguiente comando:

    ```
    python predict.py model='best.pt' source='ecuador.png' show=True
    ```

