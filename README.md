# CareerGuide GPT (Streamlit + FastAPI + LangChain + Gemini)

Remiro AI , A vishcraft Product :
- Chat interface in the main area
- Chat sessions history in the left sidebar
- Gemini model via LangChain
- Supabase Auth (email/password) for signup/login/logout
- Supabase datastore with per-user secure isolation using RLS

## API backend for Remiro frontend

This project now also exposes a dedicated AI-chat backend API for `Remiro-land_front`.

- FastAPI entrypoint: `src/api_server.py`
- Base URL (local): `http://localhost:8000`
- Endpoints:
  - `GET /api/health`
  - `GET /api/chat/sessions`
  - `POST /api/chat/sessions`
  - `PATCH /api/chat/sessions/{session_id}`
  - `GET /api/chat/sessions/{session_id}/messages`
  - `POST /api/chat/sessions/{session_id}/messages`

The API validates the same JWT bearer token produced by `Remiro-land_back` (set `JWT_SECRET` to the same value in both services).

## 1) Create and activate a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
pip install -r requirements.txt
```

## 3) Configure environment variables

Copy `.env.example` to `.env` and fill values:

- `GOOGLE_API_KEY`
- `GEMINI_MODEL` (default: `gemini-1.5-flash`)
- `SUPABASE_URL`
- `SUPABASE_KEY` (use **anon key**, not service role)

## 4) API backend environment variables

Add these to `.env` (see `.env.example`):

- `MONGODB_URI`
- `MONGODB_DB_NAME` (optional, default `remiro_ai_chat`)
- `JWT_SECRET` (must match `Remiro-land_back`)
- `ALLOWED_ORIGINS` (comma separated, e.g. `http://localhost:5173`)

## 5) Run AI chat API backend

```powershell
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

## 6) Create tables in Supabase (optional Streamlit UI path only)

Run SQL from `supabase_schema.sql` in your Supabase SQL Editor.

## 7) Configure Supabase Auth

In Supabase Dashboard:
- Go to **Authentication -> Providers -> Email** and enable Email provider
- Keep `Confirm email` ON or OFF as you prefer

If `Confirm email` is ON:
- User must verify email before login

## 8) Run the Streamlit app (optional)

```powershell
streamlit run app.py
```

## Notes

- Each authenticated user only sees their own sessions/messages.
- Access control is enforced in database policies (RLS), not only in UI code.
