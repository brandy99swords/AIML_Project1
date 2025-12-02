FROM python:3.8.5-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]

<<<<<<< HEAD:DockerFile
# update filename to Dockerfile 
=======
# update filename to Dockerfile again
>>>>>>> f783d1362b83a02d60b031688927d36a52ee3daa:Dockerfile
