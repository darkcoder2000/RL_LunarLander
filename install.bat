#python 3.7.11
pip install ray[rllib] --proxy=http://gateway.schneider.zscaler.net:80
pip install ray[tune] --proxy=http://gateway.schneider.zscaler.net:80
pip install redis --proxy=http://gateway.schneider.zscaler.net:80
pip install protobuf==3.20.0 --proxy=http://gateway.schneider.zscaler.net:80
pip install --trusted-host download.pytorch.org  torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html --proxy=http://gateway.schneider.zscaler.net:80
pip install box2d-py --proxy=http://gateway.schneider.zscaler.net:80
pip install gym[atari]==0.21.0 gym[accept-rom-license]==0.21.0 atari_py --proxy=http://gateway.schneider.zscaler.net:80
pip install stable-baselines3[extra] --proxy=http://gateway.schneider.zscaler.net:80
