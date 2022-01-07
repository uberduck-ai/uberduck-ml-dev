FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-10

WORKDIR /root

RUN apt-get update && apt-get install -y espeak libsndfile-dev

RUN pip install git+https://github.com/uberduck-ai/uberduck-ml-dev

RUN mkdir /root/bucket
RUN mkdir /root/model
RUN mkdir /root/checkpoints
RUN mkdir /root/logs

COPY secrets/gcloud_key.json /secrets/gcloud_key.json
COPY entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
