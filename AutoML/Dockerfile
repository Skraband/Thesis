FROM ubuntu:20.04

WORKDIR /automl

# install linux packages
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y ffmpeg libsm6 libxext6
#RUN apt-get install libgl1

# Set the locale
# workaround for https://github.com/automl/auto-sklearn/issues/867
#RUN apt-get -y install locales
#RUN touch /usr/share/locale/locale.alias
#RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
#ENV LANG en_US.UTF-8
#ENV LANGUAGE en_US:en
#ENV LC_ALL en_US.UTF-8

# install build requirements
RUN apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt-get install -y build-essential

RUN apt-get install -y swig

RUN pip3 install virtualenv
#RUN python3 -m venv /automl-venv
#RUN source /automl-venv/bin/activate
# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip

# Install
RUN pip3 install autoPyTorch[forecasting]

COPY test.py test.py
CMD [ "python3", "test.py"]


