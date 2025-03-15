FROM python:3.9-slim

#update and importing necessary packages
RUN apt-get update && apt-get install -y git

#installing libraries
RUN pip install --upgrade pip

#workspace setting
WORKDIR /workspace