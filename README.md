This is ringformer trained on vae bottleneck vocoder model only
in the future I will release TTS model hopefully 

pip install triton-nightly==3.0.0.post20240716052845 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/
pip install --pre torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/nightly/cu130

python train.py  -c configs/vae2_bottleneck.json  -m vae_normal
conda create -n ringformer python=3.11
