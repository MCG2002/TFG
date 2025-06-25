#!/bin/bash

# Recorre todos los archivos Python en el directorio actual
for file in test_*.py; do
    echo "==============================="
    echo "Ejecutando: $file"
    python "$file"
    echo ""
done
