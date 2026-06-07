import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    gemini_model: str
    supabase_url: str | None
    supabase_key: str | None
    serper_api_key: str | None


def get_settings() -> Settings:
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip() or None
    supabase_key = os.getenv("SUPABASE_KEY", "").strip() or None
    serper_api_key = os.getenv("SERPER_API_KEY", "").strip() or None
    missing = []
    if not google_api_key:
        missing.append("GOOGLE_API_KEY")

    if missing:
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

    return Settings(
        google_api_key=google_api_key,
        gemini_model=gemini_model,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        serper_api_key=serper_api_key,
    )
