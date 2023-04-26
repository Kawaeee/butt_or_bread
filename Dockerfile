FROM python:3.9

ENV LANG=C.UTF-8
ENV TZ=Asia/Bangkok
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

RUN wget https://github.com/Kawaeee/butt_or_bread/releases/download/v1.3/buttbread_resnet152_3.h5

ENTRYPOINT ["streamlit", "run", "/app/streamlit_app.py"]