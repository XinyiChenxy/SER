"""Shared EDA helpers."""
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, librosa, librosa.display

def plot_audio(path, ax_wave, ax_mel, sample_rate=16000):
    y,sr=librosa.load(path,sr=sample_rate,mono=True)
    times=np.arange(len(y))/sr; ax_wave.plot(times,y,lw=.6); ax_wave.set(title='Waveform',xlabel='Time (s)',ylabel='Amplitude')
    mel=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=64); db=librosa.power_to_db(mel,ref=np.max)
    img=librosa.display.specshow(db,sr=sr,x_axis='time',y_axis='mel',ax=ax_mel); ax_mel.set(title='Log-Mel spectrogram'); plt.colorbar(img,ax=ax_mel,format='%+2.0f dB')
