FROM python:3.6

ARG project_dir=/app/

WORKDIR $project_dir
RUN apt-get update \
    && apt-get install -y cmake

COPY requirements.txt $project_dir
RUN pip install -r requirements.txt --no-cache-dir

COPY . $project_dir

ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 --max-requests 1 api:app