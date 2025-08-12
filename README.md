
# Work Assistant (FastAPI + Static UI)

One service on Render hosts both the UI and the API (same origin = no CORS).

## Deploy to Render
1. Create a **Web Service** from this repo.
2. In **Environment** tab, add your secrets (e.g., `OPENAI_API_KEY`, etc.).
3. Set **Start Command** to:
   ```
   gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```
4. (Optional) Keep `.python-version` at 3.10.18 for consistency.

## Local dev
```bash
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000
```

## Env vars
- Put secrets in a local `.env` (not committed) and on Render's **Environment** tab.
- Example: see `.env.example`.

## File layout
```
work_assistant/
├─ main.py
├─ requirements.txt
├─ render.yaml
├─ .python-version
├─ .env.example
└─ static/
   ├─ index.html
   ├─ app.js
   ├─ styles.css
   └─ favicon.ico (optional)
```

## Notes
- `/chat` is a placeholder that echoes the message; wire your agent there.
- We serve `/favicon.ico` (if present) or return 204 to avoid 404 noise.
