# Vocal Emotion Lab — CREMA-D SER

Local speech-emotion recognition starter project using PyTorch, a log-Mel CNN, TensorBoard, an EDA notebook, and a Streamlit demo. Predictions are estimates of acted vocal expression, not facts about a speaker's inner state.

## Data

Dataset: [CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad). This project expects CREMA-D filenames such as `1001_DFA_ANG_XX.wav`; parser validates speaker, sentence, emotion, and intensity fields and reports unreadable/unparseable files. Six classes: angry, disgust, fear, happy, neutral, sad.

Obtain CREMA-D through Kaggle according to the dataset owner’s current terms and cite the dataset in submissions. Do not commit or redistribute the audio unless its license and course rules allow it. The audio directory is mounted read-only at `/data/crema-d`; no audio is copied into this repository.

## Build and start

On the Docker host, set the dataset path to your extracted CREMA-D directory, then build and start:

```bash
cd /home/mywsl/Workspace/MT_mid_pj
export CREMA_DATA_DIR=/home/mywsl/Workspace/datasets/CREMA-D
mkdir -p outputs
docker compose build
docker compose run --rm --service-ports ser
```

If GPU passthrough is available, add `gpus: all` under the compose service (Docker Compose supporting GPU reservations) or use `docker run --gpus all` with equivalent mounts. The included requirements install the pinned CPU-compatible PyTorch baseline; for CUDA, replace `torch`/`torchaudio` pins with the official matching CUDA wheel command for the host driver and desired PyTorch release before building. Verify inside container:

```bash
conda activate ser-emotion
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## EDA and splits

Run JupyterLab inside container with port 8888 forwarded if desired, then open `notebooks/01_cremad_eda.ipynb`. Set `DATA_DIR=/data/crema-d`. The notebook inspects labels, file health, speaker/emotion balance, durations, sample rates, channels, waveforms, FFT, log-Mel, rough silence, and speaker-independent splits.

Create manifest and invalid-file report:

```bash
python scripts/prepare_data.py --data /data/crema-d --output outputs
```

The deterministic split assigns speakers as groups (70/15/15 target); speaker independence takes priority over exact class proportions. Check `outputs/split_manifest.csv`; `outputs/invalid_files.csv` records files rejected by the parser or metadata reader.

## Train, evaluate, TensorBoard

```bash
python -m ser.train --data /data/crema-d --output outputs --epochs 50
# short pipeline smoke run (still needs the dataset mounted)
python -m ser.train --data /data/crema-d --output outputs --smoke

tensorboard --logdir outputs/runs --host 0.0.0.0 --port 6006
```

Training monitors validation macro-F1, saves `outputs/best.pt`, and evaluates test once after selection. Metrics go to `outputs/test_metrics.json`. The starter baseline uses 16 kHz mono, four-second center crop/padding and a CNN over log-Mel features. Validate split sizes/class representation before full training; this starter intentionally does not invent metrics or precomputed outputs.

## Interactive demo

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open port 8501. Upload WAV/FLAC/OGG audio to inspect waveform, log-Mel, and probabilities. Audio is handled in memory aside from a temporary local inference file, which is removed after processing. No external API is used. The generated explanation is a local template tied to model output, not an independent explanation.

## Citation, licensing, and limitations

Use the dataset page's current citation and license terms; verify these before distribution. This repository does not contain CREMA-D. Speaker-group splitting helps prevent identity leakage but does not make acted speech representative of spontaneous emotion. Confidence is not calibrated unless a separate calibration procedure is performed. Avoid psychological, clinical, hiring, or high-stakes interpretation.

## Demo script

1. Show the EDA class/speaker balance and explain speaker-level split.
2. Play a permitted sample or user-provided clip and inspect waveform/log-Mel.
3. Run the model and compare class probabilities, noting uncertainty.
4. Explain that the output estimates acted vocal expression and is not a diagnosis.
