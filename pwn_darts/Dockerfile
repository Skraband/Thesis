FROM ubuntu:20.04

WORKDIR /pwn_darts

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
#RUN apt-get install -y ffmpeg libsm6 libxext6

RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt-get install -y build-essential
RUN pip3 install --upgrade pip

ADD . .

RUN pip3 install --requirement requirements.txt
RUN pip3  install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

CMD [ "python3", "training.py"]
