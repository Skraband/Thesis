FROM ubuntu:20.04

WORKDIR /PWN

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt-get install -y build-essential
RUN pip3 install --upgrade pip

COPY requirements.txt .

RUN pip install --requirement requirements.txt

COPY . .

CMD [ "python3", "training.py"]