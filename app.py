from __future__ import annotations

import streamlit as st

from src.auth_service import AuthService
from src.config import get_settings
from src.llm_service import CareerGuideLLM
from src.repository import ChatRepository


st.set_page_config(page_title="Remiro AI - chatendpoint", layout="wide")


def short_title(text: str, words: int = 6) -> str:
    clean = " ".join(text.split())
    items = clean.split(" ")
    return " ".join(items[:words]).strip() or "New chat"


@st.cache_resource
def init_llm() -> CareerGuideLLM:
    settings = get_settings()
    llm = CareerGuideLLM(
        settings.google_api_key,
        settings.gemini_model,
        settings.serper_api_key,
    )
    return llm


def get_auth_service() -> AuthService:
    settings = get_settings()
    return AuthService(
        settings.supabase_url,
        settings.supabase_key,
        access_token=st.session_state.get("access_token"),
        refresh_token=st.session_state.get("refresh_token"),
    )


def save_auth_tokens(access_token: str | None, refresh_token: str | None) -> None:
    st.session_state.access_token = access_token
    st.session_state.refresh_token = refresh_token


def clear_auth_state() -> None:
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.selected_session_id = None


def render_auth_screen(auth: AuthService) -> None:
    st.subheader("Sign in to your account")
    login_tab, signup_tab = st.tabs(["Login", "Sign up"])

    with login_tab:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                try:
                    response = auth.sign_in(email=email, password=password)
                    access_token, refresh_token = AuthService.extract_tokens(response)
                    if not access_token or not refresh_token:
                        st.error("Login failed. Check your credentials and try again.")
                    else:
                        save_auth_tokens(access_token, refresh_token)
                        st.rerun()
                except Exception as exc:
                    st.error(f"Login error: {exc}")

    with signup_tab:
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submit = st.form_submit_button("Create account", use_container_width=True)

            if submit:
                try:
                    response = auth.sign_up(email=email, password=password)
                    access_token, refresh_token = AuthService.extract_tokens(response)
                    if access_token and refresh_token:
                        save_auth_tokens(access_token, refresh_token)
                        st.success("Account created. You are now signed in.")
                        st.rerun()
                    else:
                        st.success(
                            "Account created. If email confirmation is enabled, verify your email then log in."
                        )
                except Exception as exc:
                    st.error(f"Signup error: {exc}")


def ensure_selected_session(repo: ChatRepository) -> str:
    sessions = repo.list_sessions()

    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None

    selected = st.session_state.selected_session_id
    if selected and any(s["id"] == selected for s in sessions):
        return selected

    if sessions:
        selected = sessions[0]["id"]
        st.session_state.selected_session_id = selected
        return selected

    created = repo.create_session()
    st.session_state.selected_session_id = created["id"]
    return created["id"]


def render_sidebar(repo: ChatRepository, auth: AuthService, user_email: str) -> None:
    st.sidebar.title("Chats")
    st.sidebar.caption(f"Signed in as {user_email}")

    if st.sidebar.button("Logout", use_container_width=True):
        try:
            auth.sign_out()
        finally:
            clear_auth_state()
            st.rerun()

    if st.sidebar.button("+ New chat", use_container_width=True):
        created = repo.create_session()
        st.session_state.selected_session_id = created["id"]
        st.rerun()

    sessions = repo.list_sessions()
    selected = st.session_state.get("selected_session_id")

    if not sessions:
        st.sidebar.info("No chats yet.")
        return

    for session in sessions:
        title = (session.get("title") or "New chat").strip() or "New chat"
        session_id = session["id"]
        kind = "primary" if session_id == selected else "secondary"
        if st.sidebar.button(title, key=f"session_{session_id}", type=kind, use_container_width=True):
            st.session_state.selected_session_id = session_id
            st.rerun()


def render_messages(messages: list[dict]) -> None:
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def main() -> None:
    # Custom CSS for a clean, sleek, and modern look
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 3.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #ff007f, #7928ca);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0rem;
            padding-bottom: 0rem;
        }
        .main-subtitle {
            font-size: 1.3rem;
            font-weight: 500;
            color: #6c757d;
            margin-top: 0.5rem;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">Remiro AI - chatendpoint</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">A powerful career assistant for the bold aspirants.</div>', unsafe_allow_html=True)

    try:
        llm = init_llm()
        auth = get_auth_service()
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.info("Copy .env.example to .env and fill all required keys.")
        st.stop()

    if "access_token" not in st.session_state:
        st.session_state.access_token = None
    if "refresh_token" not in st.session_state:
        st.session_state.refresh_token = None
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = None

    user = auth.get_user()
    if not user:
        render_auth_screen(auth)
        st.stop()

    repo = ChatRepository(auth.client)
    user_email = getattr(user, "email", "user")

    selected_session_id = ensure_selected_session(repo)
    render_sidebar(repo, auth, user_email)

    messages = repo.get_messages(selected_session_id)
    render_messages(messages)

    prompt = st.chat_input("Ask for career guidance...")
    if not prompt:
        return

    repo.add_message(selected_session_id, "user", prompt)

    if len(messages) == 0:
        repo.rename_session(selected_session_id, short_title(prompt))

    updated_history = repo.get_messages(selected_session_id)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = llm.generate_reply(updated_history)
            st.markdown(reply)

    repo.add_message(selected_session_id, "assistant", reply)
    st.rerun()


if __name__ == "__main__":
    main()
