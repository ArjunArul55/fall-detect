"""
Package comprehensive project artifacts into a ZIP file.
Includes:
- All processed datasets from data/processed/ (as shown in Windows Explorer screenshot)
- All trained models (Keras models: multitask_best, cnn_lstm_best, cnn_baseline)
- All quantized edge models (TFLite: dynamic_range, float16, float32, int8)
- Microcontroller deployment C header (model_data.h)
- Preprocessing & severity scalers
- Model evaluation summaries, metadata, and Review 3 presentation files
"""

import os
import sys
import zipfile
from pathlib import Path

def create_complete_artifacts_zip():
    repo_root = Path(__file__).resolve().parent.parent
    zip_path = repo_root / "fall_detection_project_artifacts.zip"

    print("=" * 60)
    print("Packaging Fall Detection Project Artifacts...")
    print(f"Output ZIP: {zip_path}")
    print("=" * 60)

    # Specific list of processed data files matching the user's screenshot
    processed_files = [
        "data/processed/X_train_balanced.npy",
        "data/processed/y_train_balanced.npy",
        "data/processed/train.npy",
        "data/processed/train_labels.npy",
        "data/processed/train_recording_ids.npy",
        "data/processed/train_subject_ids.npy",
        "data/processed/train_activity_labels.npy",
        "data/processed/train_severity_labels.npy",
        "data/processed/val.npy",
        "data/processed/val_labels.npy",
        "data/processed/val_recording_ids.npy",
        "data/processed/val_subject_ids.npy",
        "data/processed/val_activity_labels.npy",
        "data/processed/val_severity_labels.npy",
        "data/processed/scaler.pkl",
        "data/processed/test.npy",
        "data/processed/test_labels.npy",
        "data/processed/test_recording_ids.npy",
        "data/processed/test_subject_ids.npy",
        "data/processed/test_activity_labels.npy",
        "data/processed/test_severity_labels.npy",
        "data/processed/channel_statistics.csv",
        "data/processed/class_imbalance_report.json",
        "data/processed/filter_plot.png",
        "data/processed/GYRO_OUTLIER_BY_ACTIVITY.md",
    ]

    # Model files across the project
    model_files = [
        # Production Multi-Task Model
        ("results/severity_multitask/run_v2/multitask_best.keras", "models/keras/multitask_best.keras"),
        ("results/severity_multitask/run_v2/run_metadata.json", "models/keras/multitask_metadata.json"),
        ("results/severity_multitask/run_v2/summary.md", "models/keras/multitask_summary.md"),
        
        # Baseline Models
        ("results/cnn_lstm_baseline/cnn_lstm_best.keras", "models/keras/cnn_lstm_best.keras"),
        ("results/cnn_lstm_baseline/run_metadata.json", "models/keras/cnn_lstm_metadata.json"),
        ("results/cnn_baseline/cnn_baseline_best.keras", "models/keras/cnn_baseline_best.keras"),
        
        # Edge Quantized Models
        ("results/quantization/run_v1/deployed_model.tflite", "models/quantized_tflite/deployed_model.tflite"),
        ("results/quantization/run_v1/model_dynamic_range.tflite", "models/quantized_tflite/model_dynamic_range.tflite"),
        ("results/quantization/run_v1/model_float16.tflite", "models/quantized_tflite/model_float16.tflite"),
        ("results/quantization/run_v1/model_float32.tflite", "models/quantized_tflite/model_float32.tflite"),
        ("results/quantization/run_v1/model_int8.tflite", "models/quantized_tflite/model_int8.tflite"),
        ("results/quantization/run_v1/deployed_model_meta.json", "models/quantized_tflite/deployed_model_meta.json"),
        ("results/quantization/run_v1/quantization_report.md", "models/quantized_tflite/quantization_report.md"),
        
        # Microcontroller C Header
        ("results/quantization/run_v1/model_data.h", "models/microcontroller_esp32/model_data.h"),
        
        # Model Comparison Table
        ("results/comparison/model_comparison.csv", "reports/model_comparison.csv"),
        ("results/comparison/model_comparison.md", "reports/model_comparison.md"),
        
        # Review 3 Presentation Deck & Figures
        ("docs/presentation/Review_3_Presentation.pdf", "presentation/Review_3_Presentation.pdf"),
        ("docs/presentation/Review_3_Presentation.pptx", "presentation/Review_3_Presentation.pptx"),
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Add processed dataset files
        print("\n[1/3] Adding data/processed files...")
        for rel_path in processed_files:
            file_path = repo_root / rel_path
            if file_path.exists():
                arcname = rel_path
                zf.write(file_path, arcname=arcname)
                size_kb = file_path.stat().st_size / 1024
                print(f"  + {arcname} ({size_kb:,.1f} KB)")
            else:
                print(f"  ! Missing: {rel_path}")

        # Add data/processed/severity subfolder
        severity_dir = repo_root / "data/processed/severity"
        if severity_dir.exists():
            for f in severity_dir.rglob("*"):
                if f.is_file():
                    arcname = str(f.relative_to(repo_root)).replace("\\", "/")
                    zf.write(f, arcname=arcname)
                    print(f"  + {arcname}")

        # 2. Add all model files
        print("\n[2/3] Adding model files & deployment assets...")
        for src_rel, arcname in model_files:
            src_path = repo_root / src_rel
            if src_path.exists():
                zf.write(src_path, arcname=arcname)
                size_kb = src_path.stat().st_size / 1024
                print(f"  + {arcname} ({size_kb:,.1f} KB)")
            else:
                print(f"  ! Missing: {src_rel}")

        # 3. Add root manifest & README
        manifest_text = """FALL DETECTION PROJECT - COMPLETE ARTIFACT ARCHIVE
==================================================
Institution: Madras Institute of Technology, Anna University
Course: IT5712 Project 1 (Academic Year 2023-2027)

This archive contains:
1. data/processed/
   - Filtered and windowed numpy arrays (train, val, test, balanced sets)
   - Feature scalers (scaler.pkl, severity_scaler.pkl)
   - K-Means severity clustering artifacts

2. models/
   - keras/: Dual-head multi-task CNN-LSTM, binary CNN-LSTM, 1D CNN baseline
   - quantized_tflite/: Edge TFLite models (Dynamic-Range, Float16, Float32, INT8)
   - microcontroller_esp32/: C array header (model_data.h) for embedded flash memory

3. presentation/
   - Review_3_Presentation.pptx & Review_3_Presentation.pdf

4. reports/
   - Cross-model performance comparison tables and quantization benchmarks
"""
        zf.writestr("ARCHIVE_MANIFEST.txt", manifest_text)

    total_size = zip_path.stat().st_size
    print("\n" + "=" * 60)
    print(f"Archive Created Successfully!")
    print(f"Path: {zip_path}")
    print(f"Size: {total_size:,} bytes ({total_size / (1024 * 1024):.2f} MB)")
    print("=" * 60)

if __name__ == "__main__":
    create_complete_artifacts_zip()
