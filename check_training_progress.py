"""Monitor continuo del entrenamiento YOLOv8 con GPU."""
import time
from pathlib import Path

def main():
    log_dir = Path("runs/detect/runs/train/archivev3_model_gpu3")
    
    print("=" * 80)
    print("🚀 MONITOR DE ENTRENAMIENTO - YOLO v8 con GPU")
    print("=" * 80)
    print(f"📁 Directorio: {log_dir.absolute()}")
    print()
    
    # Verificar directorio
    if not log_dir.exists():
        print("⏳ Esperando que inicie el entrenamiento...")
        print(f"   El directorio {log_dir} aún no existe.")
        return
    
    # Listar archivos generados
    print("📊 ARCHIVOS GENERADOS:")
    print("-" * 80)
    files = sorted(log_dir.glob("*"))
    for file in files:
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file.stat().st_mtime))
            print(f"  {file.name:<30} {size_mb:>8.2f} MB    {mod_time}")
        else:
            print(f"  {file.name:<30} {'<DIR>':>8}       {'':>19}")
    
    print("\n" + "=" * 80)
    
    # Resultados CSV
    results_file = log_dir / "results.csv"
    if results_file.exists():
        print("📈 MÉTRICAS DE ENTRENAMIENTO (Últimos Epochs):")
        print("-" * 80)
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Header
                    header = lines[0].strip().split(',')
                    print(f"{'Epoch':<8} {'Box Loss':<12} {'Cls Loss':<12} {'DFL Loss':<12} {'Precisión':<12}")
                    print("-" * 80)
                    
                    # Últimos 5 epochs
                    for line in lines[-5:]:
                        values = line.strip().split(',')
                        if len(values) >= 5:
                            epoch = values[0].strip()
                            box_loss = values[2].strip() if len(values) > 2 else "N/A"
                            cls_loss = values[3].strip() if len(values) > 3 else "N/A"
                            dfl_loss = values[4].strip() if len(values) > 4 else "N/A"
                            precision = values[5].strip() if len(values) > 5 else "N/A"
                            print(f"{epoch:<8} {box_loss:<12} {cls_loss:<12} {dfl_loss:<12} {precision:<12}")
        except Exception as e:
            print(f"⚠️  Error leyendo resultados: {e}")
    else:
        print("⏳ Archivo de resultados (results.csv) aún no creado")
        print("   El archivo se genera después del primer epoch completado.")
    
    print("\n" + "=" * 80)
    print("💡 COMANDOS ÚTILES:")
    print("-" * 80)
    print("  Monitor GPU:     nvidia-smi")
    print("  Ver progreso:    Get-Content results.csv -Tail 20")
    print("  Tensorboard:     tensorboard --logdir=runs/train")
    print("=" * 80)

if __name__ == '__main__':
    main()
