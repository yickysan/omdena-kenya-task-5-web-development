FROM python:3.11

WORKDIR /app

COPY ./requirements.txt requirements.txt

COPY ./setup.py setup.py

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["gunicorn", "app:app", "--host", "--port", "80"]
