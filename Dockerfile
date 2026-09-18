FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# torch CPU-only wheel first: the default PyPI build bundles CUDA (multi-GB), wasted weight on
# this CPU-only EC2 instance. Installing it from the official CPU index before the rest of
# requirements.txt means pip sees it already satisfied and doesn't pull the CUDA build instead.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
