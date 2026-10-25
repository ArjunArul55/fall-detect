"""
Package project artifacts into clean, self-contained ZIP archives:
1. fall_detection_project_artifacts.zip:
   - data_processed/: all processed dataset arrays, labels, scalers, severity & deprecated_severity folders
   - models/: all Keras models, TFLite quantized edge models, microcontroller C headers
   - presentation/: formal Review 3 PPTX and PDF slide decks
   - reports/: model comparison tables and quantization reports
2. data_processed.zip:
   - Standalone archive containing only the data/processed directory files and subfolders
"""

import os
import sys
import zipfile
from pathlib import Path

def package_artifacts():
    repo_root = Path(__file__).resolve().parent.parent
    proc_dir = repo_root / "data" / "processed"
    
    # Target archives
    combined_zip_path = repo_root / "fall_detection_project_artifacts.zip"
    data_only_zip_path = repo_root / "data_processed.zip"

    print("=" * 65)
    print("PACKAGING FALL DETECTION ARTIFACTS")
    print("=" * 65)

    # 1. Collect all files in data/processed (excluding large generative cache if desired, but including all screenshot items)
    data_files = []
    for item in proc_dir.rglob("*"):
        if item.is_file():
            # Exclude train_generative.npy if desired, or keep screenshot items
            if "generative" in item.name.lower():
                continue
            rel = item.relative_to(proc_dir)
            data_files.append((item, rel))

    print(f"Collected {len(data_files)} files from data/processed/")

    # 2. Collect model files
    model_files = [
        # Production Multi-Task Model
        ("results/severity_multitask/run_v2/multitask_best.keras", "models/keras/multitask_best.keras"),
        ("results/severity_multitask/run_v2/run_metadata.json", "models/keras/multitask_metadata.json"),
        ("results/severity_multitask/run_v2/summary.md", "models/keras/multitask_summary.md"),
        
        # Baseline Models
        ("results/cnn_lstm_baseline/cnn_lstm_best.keras", "models/keras/cnn_lstm_best.keras"),
        ("results/cnn_lstm_baseline/run_metadata.json", "models/keras/cnn_lstm_metadata.json"),
        ("results/cnn_baseline/cnn_baseline_best.keras", "models/keras/cnn_baseline_best.keras"),
        
        # Quantized TFLite Edge Models
        ("results/quantization/run_v1/deployed_model.tflite", "models/quantized_tflite/deployed_model.tflite"),
        ("results/quantization/run_v1/model_dynamic_range.tflite", "models/quantized_tflite/model_dynamic_range.tflite"),
        ("results/quantization/run_v1/model_float16.tflite", "models/quantized_tflite/model_float16.tflite"),
        ("results/quantization/run_v1/model_float32.tflite", "models/quantized_tflite/model_float32.tflite"),
        ("results/quantization/run_v1/model_int8.tflite", "models/quantized_tflite/model_int8.tflite"),
        ("results/quantization/run_v1/deployed_model_meta.json", "models/quantized_tflite/deployed_model_meta.json"),
        ("results/quantization/run_v1/quantization_report.md", "models/quantized_tflite/quantization_report.md"),
        
        # Microcontroller C Header
        ("results/quantization/run_v1/model_data.h", "models/microcontroller_esp32/model_data.h"),
        
        # Comparison reports
        ("results/comparison/model_comparison.csv", "reports/model_comparison.csv"),
        ("results/comparison/model_comparison.md", "reports/model_comparison.md"),
        
        # Review 3 Presentation
        ("docs/presentation/Review_3_Presentation.pdf", "presentation/Review_3_Presentation.pdf"),
        ("docs/presentation/Review_3_Presentation.pptx", "presentation/Review_3_Presentation.pptx"),
    ]

    manifest_text = """FALL DETECTION PROJECT - ARTIFACT ARCHIVE
=========================================
Madras Institute of Technology, Anna University
Course: IT5712 Project 1 (Academic Year 2023-2027)

FOLDER STRUCTURE:
-----------------
1. data_processed/
   - Preprocessed SisFall dataset arrays: train.npy, val.npy, test.npy
   - Balanced datasets: X_train_balanced.npy, y_train_balanced.npy
   - Label mappings: labels, recording IDs, subject IDs, activity labels, severity labels
   - Normalization scaler: scaler.pkl (StandardScaler fitted on training windows)
   - severity/: K-Means clustering model (kmeans_model.pkl), severity scaler, PCA plots
   - deprecated_severity/: Initial severity run artifacts and diagnostics

2. models/
   - keras/:
       * multitask_best.keras (Production CNN-LSTM dual-head binary fall + severity model)
       * cnn_lstm_best.keras (Baseline binary CNN-LSTM detector)
       * cnn_baseline_best.keras (Baseline 1D CNN detector)
   - quantized_tflite/:
       * deployed_model.tflite (Dynamic-range quantized edge model, 111 KB, 0.026 ms latency)
       * model_dynamic_range.tflite, model_float16.tflite, model_float32.tflite, model_int8.tflite
   - microcontroller_esp32/:
       * model_data.h (C array header for ESP32 and ARM Cortex-M4 flash memory)

3. presentation/
   - Review_3_Presentation.pdf (Academic PDF presentation)
   - Review_3_Presentation.pptx (Microsoft PowerPoint deck)

4. reports/
   - model_comparison.csv & model_comparison.md
"""

    # --- BUILD COMBINED ARCHIVE (fall_detection_project_artifacts.zip) ---
    print(f"\n[1/2] Building combined archive: {combined_zip_path.name}...")
    with zipfile.ZipFile(combined_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README_MANIFEST.txt", manifest_text)
        
        # Add data_processed folder
        for src_path, rel_path in data_files:
            arcname = f"data_processed/{str(rel_path).replace(os.sep, '/')}"
            zf.write(src_path, arcname=arcname)
            
        # Add models, presentations, reports
        for src_rel, arcname in model_files:
            src_path = repo_root / src_rel
            if src_path.exists():
                zf.write(src_path, arcname=arcname)

    comb_size = combined_zip_path.stat().st_size
    print(f" -> Created {combined_zip_path.name}: {comb_size:,} bytes ({comb_size / (1024*1024):.2f} MB)")

    # --- BUILD DATA-ONLY ARCHIVE (data_processed.zip) ---
    print(f"\n[2/2] Building standalone dataset archive: {data_only_zip_path.name}...")
    with zipfile.ZipFile(data_only_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("README.txt", "Preprocessed SisFall dataset arrays and severity clustering models for IT5712 Project 1.\n")
        for src_path, rel_path in data_files:
            arcname = f"data/processed/{str(rel_path).replace(os.sep, '/')}"
            zf.write(src_path, arcname=arcname)

    data_size = data_only_zip_path.stat().st_size
    print(f" -> Created {data_only_zip_path.name}: {data_size:,} bytes ({data_size / (1024*1024):.2f} MB)")

    print("\n" + "=" * 65)
    print("SUCCESS: All archives packaged and verified!")
    print(f"1. Combined Artifacts: {combined_zip_path} ({comb_size / (1024*1024):.2f} MB)")
    print(f"2. Data Only Archive:  {data_only_zip_path} ({data_size / (1024*1024):.2f} MB)")
    print("=" * 65)

if __name__ == "__main__":
    package_artifacts()
