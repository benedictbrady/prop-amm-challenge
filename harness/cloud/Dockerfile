FROM rust:1.85-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY harness/requirements.txt /tmp/harness-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/harness-requirements.txt

COPY harness/cloud/entrypoint.sh /usr/local/bin/harness-entrypoint
RUN chmod +x /usr/local/bin/harness-entrypoint

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/harness-entrypoint"]
