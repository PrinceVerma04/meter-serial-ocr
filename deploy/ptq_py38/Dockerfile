# Optional container path. Pinned to Python 3.8 to match the target server.
#
#   docker build -t meterocr:ptq-py38 .
#   docker run -p 8080:8080 meterocr:ptq-py38
#
# NOTE: python:3.8-slim is end-of-life upstream (3.8 reached EOL in Oct 2024),
# so it receives no further security patches. Only use this if the server's
# 3.8 requirement is fixed; otherwise build on 3.11 and drop the pins.
FROM python:3.8-slim

WORKDIR /app

# Dependencies first so the 21 MB model does not bust the layer cache.
COPY requirements-py38.txt ./
RUN pip install --no-cache-dir -r requirements-py38.txt

COPY . .

ENV METEROCR_THREADS=2 OMP_NUM_THREADS=2
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]
