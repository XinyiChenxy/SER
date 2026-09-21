# Vocal Emotion Lab — CREMA-D SER

A local Speech Emotion Recognition project using PyTorch, log-Mel features, a CNN baseline, TensorBoard, a Jupyter EDA notebook, and a Streamlit demo. The model predicts acted vocal-expression labels; it does not establish a speaker's actual emotional state.

## Dataset

Dataset: [CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad). Expected audio filenames look like `1001_DFA_ANG_XX.wav`. The parser validates speaker, sentence, emotion, and intensity tokens and writes unreadable or unparseable paths to a report. Six labels: angry, disgust, fear, happy, neutral, sad.

The dataset is not included in this repository. Place the extracted `AudioWAV` directory at the project root, or set `DATA_DIR` to its location. Follow the dataset owner's current license and citation terms; do not redistribute audio without permission.

## Create the Conda environment (WSL)

From the cloned repository:

```bash
cd ~/Workspace/SER
conda create -n ser-emotion python=3.11 pip -y
conda activate ser-emotion
python -m pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH="$PWD/src"
export DATA_DIR="$PWD/AudioWAV"
export OUTPUT_DIR="$PWD/outputs"
mkdir -p "$OUTPUT_DIR"

python -c "import torch, torchaudio; print('torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

The same pinned PyTorch/torchaudio versions are listed in `requirements.txt`. If you have an NVIDIA GPU but `torch.cuda.is_available()` is false, install the CUDA-enabled PyTorch build appropriate for your driver using the official PyTorch install selector, then rerun the check. CPU training remains available.

If your extracted audio is elsewhere, set `DATA_DIR` to the directory containing the WAV files:

```bash
export DATA_DIR="/absolute/path/to/AudioWAV"
```

## Prepare data and inspect

Generate the deterministic speaker-independent split manifest and invalid-file report:

```bash
python scripts/prepare_data.py --data "$DATA_DIR" --output "$OUTPUT_DIR"
```

Open the EDA notebook from the repository root:

```bash
jupyter lab notebooks/01_cremad_eda.ipynb
```

The notebook reports filename parsing, valid/invalid files, emotion and speaker distributions, audio metadata, duration, waveform, FFT/STFT, log-Mel features, a rough silence diagnostic, and split distributions. It does not fabricate results if the dataset path is missing.

Splitting is by speaker with a fixed seed and targets 70/15/15. Keeping speakers disjoint takes priority over exact class proportions. Verify split counts and speaker overlap in the manifest and notebook.

## Train and evaluate

```bash
python -m ser.train --data "$DATA_DIR" --output "$OUTPUT_DIR" --epochs 50
```

A short smoke run, still requiring the dataset, is:

```bash
python -m ser.train --data "$DATA_DIR" --output "$OUTPUT_DIR" --smoke
```

Training logs train and validation metrics to TensorBoard, uses validation macro-F1 for checkpoint selection and early stopping, saves the best checkpoint to `outputs/best.pt`, and evaluates the test split after model selection. Metrics are written to `outputs/test_metrics.json`. Do not use test metrics to tune the model.

In a second terminal, activate the environment, go to the repository, set `PYTHONPATH` and view TensorBoard:

```bash
cd ~/Workspace/SER
conda activate ser-emotion
export PYTHONPATH="$PWD/src"
tensorboard --logdir outputs/runs --host 0.0.0.0 --port 6006
```

## Interactive demo

After training has created `outputs/best.pt`:

```bash
cd ~/Workspace/SER
conda activate ser-emotion
export PYTHONPATH="$PWD/src"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Upload WAV, FLAC, or OGG to view its waveform, log-Mel spectrogram, and six class probabilities. The app uses a local text template for a short explanation; it does not call an external API. Uploaded audio is processed locally and is not intentionally retained or sent externally. The prediction is not a diagnosis.

## Limitations and demo flow

CREMA-D contains acted speech from a limited pool of speakers; results may not transfer to spontaneous speech or new recording conditions. Confidence scores are not calibrated unless a calibration step is added. Avoid clinical or other high-stakes interpretations.

For a demo: show class/speaker balance in EDA, play an allowed or user-provided clip, inspect waveform/log-Mel, compare predicted class probabilities, and explain the limits of acted-speech classification.
