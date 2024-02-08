FROM python:3.8

ARG MODE_DEPLOY
ARG TAG
ARG PORT_SERVER
ENV WORKERS=1
ENV THREADS=2
ENV TIMEOUT=0
ENV PYTHONUNBUFFERED True

ADD / /app

WORKDIR /app
RUN ls -lf

RUN apt update && apt install ffmpeg libsm6 libxext6  -y

# Python dependencies
RUN pip install -r /app/requirements.txt

RUN rm -rf /tmp/* /var/tmp/*
RUN rm -rf /app/.git/*

CMD exec gunicorn --bind :$PORT_SERVER --workers $WORKERS --threads $THREADS --timeout $TIMEOUT --preload main:app
