FROM python:3.11

WORKDIR /app

COPY ./requirements.txt requirements.txt

COPY . /app

RUN pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["gunicorn", "app:app", "--host", "--port", "80"]
