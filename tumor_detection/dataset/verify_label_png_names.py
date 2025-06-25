from pathlib import Path

def compare_names_between_pngs_and_txts(folder1, folder2):
    """
    Función destinada a la comparación de los nombres de archivos (sin extensión) entre dos carpetas.
    Imprime si son iguales o muestra las diferencias.
    """
    folder1 = Path(folder1)
    folder2 = Path(folder2)

    files1 = {f.stem for f in folder1.glob("*") if f.is_file()}
    files2 = {f.stem for f in folder2.glob("*") if f.is_file()}

    only_in_1 = files1 - files2
    only_in_2 = files2 - files1

    if not only_in_1 and not only_in_2:
        print("Todos los archivos coinciden entre ambas carpetas.")
        return True
    else:
        print("Hay diferencias entre las carpetas:")
        if only_in_1:
            print(f"Solo en {folder1.name}:")
            for f in sorted(only_in_1):
                print(f"  - {f}")
        if only_in_2:
            print(f"Solo en {folder2.name}:")
            for f in sorted(only_in_2):
                print(f"  - {f}")
        return False


if __name__ == "__main__":
    compare_names_between_pngs_and_txts(
        "../../dataset/manifest/CBIS-DDSM/Mass_Test/ROI mask images/TODAS_JUNTAS(YOLO)/labels",
        "../../dataset/manifest/CBIS-DDSM/Mass_Test/full mammogram images/TODAS_JUNTAS(YOLO)/labels"
    )

    compare_names_between_pngs_and_txts(
        "../../dataset/manifest/CBIS-DDSM/Mass_Training/ROI mask images/TODAS_JUNTAS(YOLO)/labels",
        "../../dataset/manifest/CBIS-DDSM/Mass_Training/full mammogram images/TODAS_JUNTAS(YOLO)/labels"
    )