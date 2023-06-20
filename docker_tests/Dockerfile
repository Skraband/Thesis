FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /docker_test

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
#RUN apt-get install -y ffmpeg libsm6 libxext6

RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt-get install -y build-essential
RUN pip3 install --upgrade pip

RUN pip3  install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

COPY check_smape.py .
ADD test_data_folder test_data_folder


CMD [ "python3", "check_smape.py"]
