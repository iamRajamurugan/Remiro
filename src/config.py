import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str
    supabase_url: str
    supabase_key: str
    serper_api_key: str | None


def get_settings() -> Settings:
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()
    serper_api_key = os.getenv("SERPER_API_KEY", "").strip() or None
    missing = []
    if not google_api_key:
        missing.append("GOOGLE_API_KEY")
    if not supabase_url:
        missing.append("SUPABASE_URL")
    if not supabase_key:
        missing.append("SUPABASE_KEY")

    if missing:
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

    return Settings(
        google_api_key=google_api_key,
        gemini_model=gemini_model,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        serper_api_key=serper_api_key,
    )
