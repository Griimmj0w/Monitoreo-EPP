"""Monitor the training progress of YOLOv8."""
import time
import os
from pathlib import Path

def monitor_training():
    log_dir = Path("runs/detect/runs/train/archivev3_model_gpu3")
    results_file = log_dir / "results.csv"
    
    print("=" * 80)
    print("MONITOR DE ENTRENAMIENTO - GPU ACTIVADA")
    print("=" * 80)
    print(f"Directorio: {log_dir}")
    print(f"Archivo de resultados: {results_file}")
    print()
    
    if not log_dir.exists():
        print("❌ No se encuentra el directorio de entrenamiento")
        return
    
    print("📊 Archivos generados:")
    for file in sorted(log_dir.glob("*")):
        size = file.stat().st_size if file.is_file() else "-"
        print(f"  - {file.name:<30} {size:>12}")
    
    print("\n" + "=" * 80)
    
    if results_file.exists():
        print("\n📈 ÚLTIMOS RESULTADOS:")
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                # Mostrar header
                print(lines[0].strip())
                # Mostrar últimas 5 líneas
                for line in lines[-5:]:
                    print(line.strip())
            else:
                print("⏳ Esperando resultados...")
    else:
        print("⏳ Archivo de resultados aún no creado")
    
    print("\n" + "=" * 80)
    print("💡 Comandos útiles:")
    print("  - Ver GPU usage: nvidia-smi")
    print("  - Ver logs completos: Get-Content runs/detect/runs/train/archivev3_model_gpu3/results.csv")
    print("=" * 80)

if __name__ == '__main__':
    monitor_training()
