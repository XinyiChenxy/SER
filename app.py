"""Local Streamlit SER demo. Uploads are kept in memory and not persisted."""
import io, os
import numpy as np
import streamlit as st
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from ser.model import SERModel
from ser.data import EMOTIONS

st.set_page_config(page_title='Vocal Emotion Lab',page_icon='🎙️',layout='wide')
st.markdown('''<style>
:root{color-scheme:dark} .stApp{background:linear-gradient(145deg,#101827,#1e293b 55%,#312e49);color:#eef2ff}
[data-testid="stMetric"]{background:#ffffff0c;border:1px solid #ffffff1c;padding:18px;border-radius:16px}
.hero{padding:28px;border:1px solid #ffffff22;border-radius:20px;background:linear-gradient(120deg,#27344c,#463657);margin-bottom:20px}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Vocal Emotion Lab</h1><p>Explore how vocal acoustics relate to a model’s emotion prediction.</p></div>',unsafe_allow_html=True)
@st.cache_resource
def load_model(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=False); model=SERModel(); model.load_state_dict(ckpt['model_state']); model.eval(); return model,ckpt
model_path=os.environ.get('SER_CHECKPOINT','outputs/best.pt')
file=st.file_uploader('Upload a short audio clip',type=['wav','flac','ogg'])
if file:
    audio_bytes=file.getvalue(); st.audio(audio_bytes)
    try:
        y,sr=sf.read(io.BytesIO(audio_bytes),always_2d=True); y=y.mean(axis=1)
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(8,2.5)); ax.plot(np.arange(len(y))/sr,y,lw=.6,color='#9b8cff'); ax.set(xlabel='Time (s)',ylabel='Amplitude',title='Waveform'); st.pyplot(fig); plt.close(fig)
        with c2:
            import librosa, librosa.display
            yy=librosa.resample(y,orig_sr=sr,target_sr=16000) if sr!=16000 else y
            mel=librosa.feature.melspectrogram(y=yy,sr=16000,n_mels=64); db=librosa.power_to_db(mel,ref=np.max)
            fig,ax=plt.subplots(figsize=(8,2.5)); im=librosa.display.specshow(db,sr=16000,x_axis='time',y_axis='mel',ax=ax,cmap='magma'); ax.set_title('Log-Mel spectrogram'); fig.colorbar(im,ax=ax,format='%+2.0f dB'); st.pyplot(fig); plt.close(fig)
        if not os.path.exists(model_path): st.info('Train a model first. No checkpoint is available yet; no prediction has been fabricated.')
        else:
            from ser.data import load_audio
            m,ckpt=load_model(model_path); tmp='/tmp/ser_upload.wav'; sf.write(tmp,y,sr); x=load_audio(tmp,ckpt['sample_rate'],ckpt['duration']).unsqueeze(0)
            with torch.inference_mode(): probs=torch.softmax(m(x),dim=1)[0].numpy()
            labels=ckpt['labels']; idx=int(probs.argmax()); st.subheader(f'Model prediction: {labels[idx]}')
            st.progress(float(probs[idx]),text=f'Confidence score: {probs[idx]:.1%}')
            chart={labels[i]:float(probs[i]) for i in range(len(labels))}; st.bar_chart(chart)
            if probs[idx]<.45: st.warning('Low confidence: the model does not distinguish one class clearly.')
            else: st.success('One vocal-expression class received the highest model score.')
            st.caption('Generated explanation (local template): The model assigned its highest score to “%s” based on learned acoustic patterns. This is a prediction, not a diagnosis or a confirmed emotional state.'%labels[idx])
            os.remove(tmp)
    except Exception as e: st.error(f'Could not process this audio: {e}')
else: st.info('Upload a supported audio file to inspect its waveform and vocal-expression prediction.')
st.caption('Privacy: this demo processes the uploaded audio locally in this container and does not save or transmit it intentionally. Model predictions are uncertain and are not psychological diagnoses.')
