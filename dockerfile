FROM python:3.7.6

#path of the network files
ENV PYTHONPATH=/home/simone/RadioML/CNN/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#directory correspondent to OCCAM destination
RUN mkdir /archive && mkdir /archive/home && mkdir /archive/home/sammazza && mkdir /archive/home/sammazza/radioML && mkdir /archive/home/sammazza/radioML/data

RUN mkdir /run_CNN
WORKDIR /run_CNN

ADD CNN.py /run_CNN/CNN.py
ADD utility/image_provider.py /run_CNN/utility/image_provider.py
ADD utility/network.py /run_CNN/utility/network.py

ENTRYPOINT ["python","/run_CNN/CNN.py"]
