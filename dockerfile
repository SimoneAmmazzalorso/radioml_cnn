FROM python:3.7.6

ENV PYTHONPATH=/home/simone/RadioML/CNN/simone/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir /archive && mkdir /archive/home && mkdir /archive/home/sammazza && mkdir /archive/home/sammazza/radioML && mkdir /archive/home/sammazza/radioML/data

RUN mkdir /run_CNN
WORKDIR /run_CNN

ADD CNN.py /run_CNN/CNN.py
ADD utility/__init__.py /run_CNN/utility/__init__.py
ADD utility/image_provider.py /run_CNN/utility/image_provider.py
ADD utility/network.py /run_CNN/utility/network.py

#ADD saved_model /saved_model
#ADD mapsim_tif /mapsim_tif

CMD python /run_CNN/CNN.py
