# ─────────────────────────────────────────────────────────────
# Dockerfile for AWS Lambda (Python 3.10) – ASL prediction API
# ─────────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.10

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artefacts
COPY app ./app
COPY models/model.joblib  .
COPY models/encoder.joblib .

# Expose paths as env-vars so main.py can load them
ENV MODEL_PATH=/var/task/model.joblib
ENV ENCODER_PATH=/var/task/encoder.joblib

# Default Lambda handler path
CMD ["app.main.lambda_handler"]
