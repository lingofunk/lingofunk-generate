FROM python:3.6-jessie

COPY . /app
WORKDIR /app

RUN pip install -r ./requirements.txt
RUN cp -r models_bank/models_char models
