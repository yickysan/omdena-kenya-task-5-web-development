From python 3.11

WORKDIR /app

COPY ./requirements.txt /MLOPS/requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

CMD python ./app.py