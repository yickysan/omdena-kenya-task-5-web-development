FROM python: 3.11

WORKDIR /app

COPY ./requirements.txt /omdena-kenya-task-5-web-development/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

CMD ["gunicorn", "app:app"]
