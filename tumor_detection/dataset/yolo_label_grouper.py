from pathlib import Path
import shutil
from collections import defaultdict

def grouping_and_renaming_labels(carpeta_labels, eliminar_originales=True):
    """ 
    Función destinada, dada una carpeta con archivos .txt, a la agrupación en función de los nombres de los archivos 
    prescindiendo del número final puesto que hacen referencia a la misma mama.
    De esta forma, realiza una de las siguientes acciones para cada grupo:
    - Si hay más de un archivo en el grupo, los combina en un solo archivo de salida.
    - Si solo hay un archivo en el grupo, lo copia sin cambios a un archivo de salida.
    Los archivos originales se eliminan después de ser procesados, si el parámetro 'eliminar_originales' es True.
    La función imprime mensajes sobre el progreso de cada acción: combinación, copia o eliminación.
    """
    carpeta = Path(carpeta_labels)
    archivos = sorted([f for f in carpeta.glob("*.txt") if f.is_file()])

    grupos = defaultdict(list)

    for archivo in archivos:
        nombre = archivo.stem  # sin .txt
        partes = nombre.split("_")
        if partes[-1].isdigit():
            clave = "_".join(partes[:-1])
            grupos[clave].append(archivo)

    for clave, lista_archivos in grupos.items():
        archivo_salida = carpeta / f"{clave}.txt"

        if len(lista_archivos) > 1:
            with open(archivo_salida, "w") as salida:
                for archivo in sorted(lista_archivos):
                    with open(archivo, "r") as f:
                        salida.writelines(f.readlines())
            print(f"Combinado: {archivo_salida.name} ({len(lista_archivos)} archivos)")

        elif len(lista_archivos) == 1:
            shutil.copy(lista_archivos[0], archivo_salida)
            print(f"Copiado: {lista_archivos[0].name} ➜ {archivo_salida.name}")

        if eliminar_originales:
            for archivo in lista_archivos:
                archivo.unlink()
                print(f"Eliminado: {archivo.name}")

if __name__ == "__main__":
    label_folder_path_val = "../../dataset/manifest/CBIS-DDSM/Mass_Test/ROI mask images/TODAS_JUNTAS(YOLO)/labels"
    grouping_and_renaming_labels(label_folder_path_val, eliminar_originales=True)

    label_folder_path_train = "../../dataset/manifest/CBIS-DDSM/Mass_Test/ROI mask images/TODAS_JUNTAS(YOLO)/labels"
    grouping_and_renaming_labels(label_folder_path_train, eliminar_originales=True)