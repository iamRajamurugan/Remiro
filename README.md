# CareerGuide GPT (Streamlit + LangChain + Gemini + Supabase)

Remiro AI , A vishcraft Product :
- Chat interface in the main area
- Chat sessions history in the left sidebar
- Gemini model via LangChain
- Supabase Auth (email/password) for signup/login/logout
- Supabase datastore with per-user secure isolation using RLS

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

## 4) Create tables in Supabase

Run SQL from `supabase_schema.sql` in your Supabase SQL Editor.

## 5) Configure Supabase Auth

In Supabase Dashboard:
- Go to **Authentication -> Providers -> Email** and enable Email provider
- Keep `Confirm email` ON or OFF as you prefer

If `Confirm email` is ON:
- User must verify email before login

## 6) Run the app

```powershell
streamlit run app.py
```

## Notes

- Each authenticated user only sees their own sessions/messages.
- Access control is enforced in database policies (RLS), not only in UI code.
