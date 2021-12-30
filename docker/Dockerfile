FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-10

WORKDIR /root

RUN apt-get update && apt-get install -y espeak libsndfile-dev
RUN mkdir /root/bucket

COPY secrets/gcloud_key.json /secrets/gcloud_key.json
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

RUN pip install git+https://github.com/uberduck-ai/uberduck-ml-dev

ENTRYPOINT ["./entrypoint.sh"]
