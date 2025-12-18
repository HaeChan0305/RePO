pip install psutil
pip install --no-cache-dir --no-build-isolation flash-attn==2.7.4.post1
pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 && pip install flash-attn==2.7.4.post1
pip install -r requirements.txt