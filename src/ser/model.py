"""Log-Mel CNN baseline."""
import torch
from torch import nn
import torchaudio

class SERModel(nn.Module):
    def __init__(self, sample_rate: int=16000, n_mels: int=64, num_classes: int=6):
        super().__init__()
        self.mel=torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_fft=1024,hop_length=256,n_mels=n_mels)
        self.db=torchaudio.transforms.AmplitudeToDB()
        self.features=nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(.3),nn.Linear(128,num_classes))
    def forward(self,x):
        if x.ndim==3: x=x.squeeze(1)
        return self.features(self.db(self.mel(x)).unsqueeze(1))
