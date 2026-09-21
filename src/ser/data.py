"""CREMA-D discovery, filename parsing, speaker-group splits, and audio loading."""
from __future__ import annotations
import re, random
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.model_selection import GroupShuffleSplit

EMOTIONS = {"ANG":"angry", "DIS":"disgust", "FEA":"fear", "HAP":"happy", "NEU":"neutral", "SAD":"sad"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}
# Typical CREMA-D basename: 1001_DFA_ANG_XX.wav. Validate token positions and known labels.
NAME_RE = re.compile(r"^(?P<speaker>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_(?P<intensity>[A-Z]{2})$", re.I)

def parse_filename(path: Path) -> dict:
    match = NAME_RE.fullmatch(path.stem)
    if not match: raise ValueError(f"Unexpected CREMA-D filename: {path.name}")
    d = match.groupdict(); code=d["emotion"].upper()
    return {"speaker_id":d["speaker"], "sentence":d["sentence"].upper(), "emotion_code":code, "emotion":EMOTIONS[code], "intensity":d["intensity"].upper()}

def scan_dataset(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root=Path(root).expanduser()
    if not root.exists(): raise FileNotFoundError(f"CREMA-D not found at {root}. Mount the dataset and set DATA_DIR.")
    rows=[]; bad=[]
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in AUDIO_EXTS):
        try:
            meta=parse_filename(p); info=sf.info(str(p))
            if info.frames <= 0 or info.samplerate <= 0: raise ValueError("Empty/invalid audio")
            meta.update(path=str(p), relative_path=str(p.relative_to(root)), duration=info.duration, sample_rate=info.samplerate, channels=info.channels, subtype=info.subtype)
            rows.append(meta)
        except Exception as e: bad.append({"path":str(p), "error":str(e)})
    return pd.DataFrame(rows), pd.DataFrame(bad, columns=["path","error"])

def make_splits(df: pd.DataFrame, seed: int=42, train_ratio: float=.70, val_ratio: float=.15) -> pd.DataFrame:
    if df.empty: raise ValueError("No valid audio records to split")
    groups=df.speaker_id.to_numpy(); y=df.emotion.to_numpy(); idx=np.arange(len(df))
    gss=GroupShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
    tr, rest=next(gss.split(idx,y,groups)); rest_df=df.iloc[rest]
    relative_val=val_ratio/(1-train_ratio)
    gss2=GroupShuffleSplit(n_splits=1,test_size=1-relative_val,random_state=seed+1)
    va_rel, te_rel=next(gss2.split(np.arange(len(rest_df)),rest_df.emotion,rest_df.speaker_id))
    split=np.full(len(df),"train",dtype=object); split[rest[va_rel]]="validation"; split[rest[te_rel]]="test"
    result=df.copy(); result["split"]=split
    sets=[set(result.loc[result.split==s,"speaker_id"]) for s in ("train","validation","test")]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise RuntimeError("Speaker leakage detected")
    return result

def load_audio(path: str | Path, sample_rate: int=16000, duration: float=4.0) -> torch.Tensor:
    waveform, sr=torchaudio.load(str(path)); waveform=waveform.mean(dim=0,keepdim=True)
    if sr != sample_rate: waveform=torchaudio.functional.resample(waveform,sr,sample_rate)
    target=int(sample_rate*duration); n=waveform.shape[-1]
    if n>target:
        start=max((n-target)//2,0); waveform=waveform[:,start:start+target]
    elif n<target: waveform=torch.nn.functional.pad(waveform,(0,target-n))
    return waveform
