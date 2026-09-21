FROM continuumio/miniconda3:24.3.0-0
WORKDIR /workspace/MT_mid_pj
RUN conda create -y -n ser-emotion python=3.11 pip && conda clean -afy
SHELL ["conda", "run", "-n", "ser-emotion", "/bin/bash", "-c"]
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PATH=/opt/conda/envs/ser-emotion/bin:$PATH \
    PYTHONPATH=/workspace/MT_mid_pj/src \
    MPLBACKEND=Agg \
    DATA_DIR=/data/crema-d \
    OUTPUT_DIR=/workspace/MT_mid_pj/outputs
EXPOSE 8501 6006
CMD ["bash"]
