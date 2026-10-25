"""Package model artifacts into a zip file for email attachment."""
import os
import zipfile
from pathlib import Path

def create_model_artifacts_zip():
    repo_root = Path(__file__).resolve().parent.parent
    zip_output_path = repo_root / "model_artifacts.zip"

    files_to_pack = [
        # Quantized TFLite edge models
        ("results/quantization/run_v1/deployed_model.tflite", "tflite_quantized/deployed_model.tflite"),
        ("results/quantization/run_v1/model_dynamic_range.tflite", "tflite_quantized/model_dynamic_range.tflite"),
        ("results/quantization/run_v1/model_float16.tflite", "tflite_quantized/model_float16.tflite"),
        ("results/quantization/run_v1/model_float32.tflite", "tflite_quantized/model_float32.tflite"),
        ("results/quantization/run_v1/model_int8.tflite", "tflite_quantized/model_int8.tflite"),
        
        # Microcontroller C array header
        ("results/quantization/run_v1/model_data.h", "microcontroller_c_header/model_data.h"),
        
        # Keras deep learning models
        ("results/severity_multitask/run_v2/multitask_best.keras", "keras_models/multitask_best.keras"),
        ("results/cnn_lstm_baseline/cnn_lstm_best.keras", "keras_models/cnn_lstm_best.keras"),
        ("results/cnn_baseline/cnn_baseline_best.keras", "keras_models/cnn_baseline_best.keras"),
        
        # Preprocessing Scalers & Clustering
        ("data/processed/scaler.pkl", "scalers/scaler.pkl"),
        ("data/processed/severity/runs/k2_20260914T055946Z/severity_scaler.pkl", "scalers/severity_scaler.pkl"),
        ("data/processed/severity/runs/k2_20260914T055946Z/kmeans_model.pkl", "scalers/kmeans_model.pkl"),
        
        # Metadata and Reports
        ("results/quantization/run_v1/deployed_model_meta.json", "metadata/deployed_model_meta.json"),
        ("results/quantization/run_v1/quantization_report.md", "metadata/quantization_report.md"),
        ("results/severity_multitask/run_v2/run_metadata.json", "metadata/multitask_run_metadata.json"),
        ("results/severity_multitask/run_v2/summary.md", "metadata/multitask_summary.md"),
        ("results/comparison/model_comparison.csv", "metadata/model_comparison.csv"),
    ]

    readme_content = """FALL DETECTION PROJECT - MODEL ARTIFACTS
Project: IT5712 Project 1, Anna University (MIT Campus)
Dataset: SisFall IMU Dataset (200 Hz downsampled to 20 Hz, 64-sample windows)

Included Artifacts:
==================
1. tflite_quantized/
   - deployed_model.tflite (111 KB Dynamic-Range quantized model, 0.026 ms latency on CPU, 0.9950 recall)
   - model_dynamic_range.tflite, model_float16.tflite, model_float32.tflite, model_int8.tflite

2. microcontroller_c_header/
   - model_data.h (113,704-byte C unsigned char array ready for ESP32 and ARM Cortex-M4 flash memory)

3. keras_models/
   - multitask_best.keras (Dual-head CNN-LSTM for binary fall detection + 2-class severity classification)
   - cnn_lstm_best.keras (Baseline CNN-LSTM binary detector)
   - cnn_baseline_best.keras (Baseline 1D CNN detector)

4. scalers/
   - scaler.pkl (StandardScaler fitted on training windows)
   - severity_scaler.pkl & kmeans_model.pkl (K-Means severity clustering artifacts)

5. metadata/
   - Metrics, architecture parameters, and performance comparison tables
"""

    print(f"Creating archive: {zip_output_path}")
    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme_content)
        for src_rel, dst_rel in files_to_pack:
            src_path = repo_root / src_rel
            if src_path.exists():
                zf.write(src_path, arcname=dst_rel)
                print(f" Added: {dst_rel} ({src_path.stat().st_size:,} bytes)")
            else:
                print(f" Warning: File not found: {src_rel}")

    zip_size_bytes = zip_output_path.stat().st_size
    print(f"\nSuccessfully created {zip_output_path.name}: {zip_size_bytes:,} bytes ({zip_size_bytes / (1024 * 1024):.2f} MB)")
    return zip_output_path

if __name__ == "__main__":
    create_model_artifacts_zip()
