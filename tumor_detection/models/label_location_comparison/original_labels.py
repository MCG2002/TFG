import cv2

def draw_box_yolo_style(
    img, 
    x1, y1, x2, y2, 
    label="tumor", 
    score=0.0, 
    font_scale=0.8, 
    font_thickness=2,
    color=(0, 0, 255),
    min_box_size=50
):
    """
    Función destinada a la creacción sobre la imagen de una caja basándose en el estilo de YOLO, es decir, un rectángulo rojo 
    señalando la zona afectada con un pequeño recuadro encima de él que continene la etiqueta que indica la puntuación sobre 
    cuánta exactitud posee la posición del rectángulo predicho por el modelo respecto a la posición real de éste.
    """

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Dibuja el rectángulo principal
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < min_box_size or box_h < min_box_size:
        thickness = 3
    else:
        thickness = 2

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    # Crea el texto calculando además su tamaño
    text = f"{label} {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Posiciona la etiqueta
    text_x = x1
    text_y = y1 - 5

    img_h, img_w = img.shape[:2]
    top_rect_y = text_y - text_h - baseline
    if top_rect_y < 0:
        text_y = y1 + text_h + baseline + 5
        top_rect_y = y1

    # Dibuja el recuadro relleno para el texto
    cv2.rectangle(
        img,
        (text_x, top_rect_y),
        (text_x + text_w, text_y),
        color,
        -1
    )

    # Posiciona el texto en blanco encima
    cv2.putText(
        img,
        text,
        (text_x, text_y - baseline),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )


def draw_yolo_labels(image_path, label_path):
    """
    Función destinada a la visualización de las mamografías destacando la localización de los tumores existentes  
    simulando lo mejor posible el estilo propio del output de YOLO. 
    """
    # Lee la imagen .png
    img = cv2.imread(image_path)
    if img is None:
        print("No se pudo cargar la imagen:", image_path)
        return

    h, w = img.shape[:2]

    # Lee el archivo .txt con las etiquetas YOLO asociado a la imagen tomada anteriormente
    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except:
        print("No se pudo leer el archivo de etiquetas:", label_path)
        return

    # Obtiene la información procedente de las etiquetas cuyo formato es: (class_id x_center y_center w_norm h_norm)
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center_norm, y_center_norm, w_norm, h_norm = parts
        x_center_norm = float(x_center_norm)
        y_center_norm = float(y_center_norm)
        w_box_norm = float(w_norm)
        h_box_norm = float(h_norm)

        # Convierte a píxeles reales
        x_center = x_center_norm * w
        y_center = y_center_norm * h
        w_box = w_box_norm * w
        h_box = h_box_norm * h

        x1 = x_center - (w_box / 2)
        y1 = y_center - (h_box / 2)
        x2 = x_center + (w_box / 2)
        y2 = y_center + (h_box / 2)

        # Dibuja los rectángulos propios eel estilo YOLO
        draw_box_yolo_style(
            img, 
            x1, y1, x2, y2, 
            label="tumor", 
            score=1.0,
            font_scale=3,
            font_thickness=14
        )

    # Visualiza los resultados
    import matplotlib.pyplot as plt
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
