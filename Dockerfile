FROM python:3.8-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

VOLUME /graphs

RUN mkdir /graphs

# Copiar dataset 
COPY shopping_trends_updated.csv .

EXPOSE 8000

WORKDIR /app
# Definir punto de entrada
ENTRYPOINT ["python", "main.py"] && tail -f /dev/null
