#!/bin/sh
# Downloads the BKT checkpoint from GCS on container startup.
# Required env var: GCS_CHECKPOINT_URI  e.g. gs://your-bucket/model.pt
set -e

if [ -z "$GCS_CHECKPOINT_URI" ]; then
  echo "ERROR: GCS_CHECKPOINT_URI is not set" >&2
  exit 1
fi

echo "Downloading checkpoint from ${GCS_CHECKPOINT_URI}..."
python - <<EOF
from google.cloud import storage

uri = "${GCS_CHECKPOINT_URI}"          # gs://bucket/path/to/model.pt
bucket_name, blob_path = uri[5:].split("/", 1)

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)
blob.download_to_filename("/app/model.pt")
print("Checkpoint downloaded to /app/model.pt")
EOF
