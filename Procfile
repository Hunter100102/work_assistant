web: gunicorn -k uvicorn.workers.UvicornWorker main:app \
  --bind 0.0.0.0:${PORT:-10000} \
  --workers ${WEB_CONCURRENCY:-1} \
  --threads ${THREADS:-2} \
  --timeout ${TIMEOUT:-120} \
  --keep-alive 25 \
  --max-requests 200 --max-requests-jitter 50
