from html import escape

import streamlit as st

from bedrock_chat import create_retrieval_qa, get_answer


st.set_page_config(
    page_title="Sharesource Ops Support",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_message(sender: str, content: str) -> None:
    role = "user" if sender == "User" else "assistant"
    safe_content = escape(content).replace("\n", "<br>")
    initials = "YU" if role == "user" else "AI"

    st.markdown(
        f"""
        <div class="message-row {role}">
            <div class="message-avatar">{initials}</div>
            <div class="message-card">
                <div class="message-label">{sender}</div>
                <div class="message-content">{safe_content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa" not in st.session_state or "memory" not in st.session_state:
    qa, memory = create_retrieval_qa()
    st.session_state.qa = qa
    st.session_state.memory = memory


st.markdown(
    """
    <style>
    :root {
        --bg: #f6eef8;
        --panel: rgba(255, 255, 255, 0.9);
        --panel-strong: #ffffff;
        --text: #31183b;
        --muted: #755d80;
        --line: rgba(93, 43, 111, 0.12);
        --brand: #8a2d80;
        --brand-mid: #a03d90;
        --brand-deep: #612264;
        --brand-soft: #f1dff3;
        --shadow: 0 22px 60px rgba(84, 31, 96, 0.14);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(138, 45, 128, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(170, 67, 140, 0.16), transparent 24%),
            linear-gradient(180deg, #fbf7fc 0%, var(--bg) 52%, #f1e4f3 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(251, 247, 252, 0.84);
        border-right: 1px solid rgba(138, 45, 128, 0.12);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        margin-bottom: 1.4rem;
        border: 1px solid rgba(255, 255, 255, 0.65);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(110, 31, 111, 0.98), rgba(143, 43, 129, 0.95) 58%, rgba(103, 31, 110, 0.92)),
            linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
        box-shadow: var(--shadow);
        color: #f8fafc;
    }

    .hero-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        inset: auto -60px -60px auto;
        width: 220px;
        height: 220px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
    }

    .brand-logo {
        width: 188px;
        max-width: 100%;
        height: auto;
        border-radius: 14px;
        box-shadow: 0 12px 28px rgba(48, 14, 56, 0.22);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 1rem 0 0.45rem;
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.05;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        max-width: 760px;
        margin: 0;
        color: rgba(248, 250, 252, 0.86);
        font-size: 1rem;
        line-height: 1.65;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin-bottom: 1rem;
    }

    .stat-card,
    .sidebar-card {
        padding: 1rem 1.05rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--panel);
        backdrop-filter: blur(12px);
        box-shadow: 0 14px 40px rgba(148, 163, 184, 0.12);
    }

    .stat-label,
    .sidebar-label {
        display: block;
        margin-bottom: 0.35rem;
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .stat-value {
        color: var(--text);
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .stat-note,
    .sidebar-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.55;
    }

    .chat-shell {
        padding: 1.15rem;
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.62);
        backdrop-filter: blur(16px);
        box-shadow: var(--shadow);
    }

    .section-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 0.2rem 0.4rem 1rem;
    }

    .section-title {
        margin: 0;
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .section-copy {
        margin: 0.2rem 0 0;
        color: var(--muted);
        font-size: 0.95rem;
    }

    .section-tag {
        padding: 0.45rem 0.7rem;
        border-radius: 999px;
        background: rgba(138, 45, 128, 0.12);
        color: var(--brand-deep);
        font-size: 0.8rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .empty-state {
        padding: 2rem 1.2rem;
        border: 1px dashed rgba(95, 107, 133, 0.28);
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.56);
        text-align: center;
    }

    .empty-title {
        margin: 0 0 0.45rem;
        color: var(--text);
        font-size: 1.2rem;
        font-weight: 700;
    }

    .empty-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .message-row {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
        margin: 0.85rem 0;
    }

    .message-row.user {
        flex-direction: row-reverse;
    }

    .message-avatar {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(138, 45, 128, 0.18), rgba(138, 45, 128, 0.08));
        color: var(--brand-deep);
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }

    .message-row.user .message-avatar {
        background: linear-gradient(135deg, rgba(199, 109, 174, 0.22), rgba(233, 192, 226, 0.12));
        color: #7a226d;
    }

    .message-card {
        width: min(78%, 840px);
        padding: 1rem 1.05rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        background: var(--panel-strong);
        box-shadow: 0 12px 30px rgba(148, 163, 184, 0.1);
    }

    .message-row.user .message-card {
        background: linear-gradient(135deg, #f9edf8, #fff9fe);
    }

    .message-label {
        margin-bottom: 0.35rem;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .message-content {
        color: var(--text);
        font-size: 0.98rem;
        line-height: 1.7;
    }

    .input-shell {
        margin-top: 1rem;
        padding: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.7);
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 34px rgba(148, 163, 184, 0.12);
    }

    .stTextInput label {
        color: var(--muted) !important;
        font-weight: 600 !important;
    }

    .stTextInput > div > div > input {
        height: 3.2rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(248, 250, 252, 0.92);
        color: var(--text);
        font-size: 0.98rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(138, 45, 128, 0.6);
        box-shadow: 0 0 0 1px rgba(138, 45, 128, 0.18);
    }

    .stButton > button,
    .stFormSubmitButton > button {
        min-height: 3.2rem;
        border: none;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--brand), var(--brand-deep));
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 14px 28px rgba(138, 45, 128, 0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 32px rgba(138, 45, 128, 0.28);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.2rem;
        }

        .hero-card {
            padding: 1.4rem;
        }

        .hero-topbar {
            flex-direction: column;
            align-items: flex-start;
        }

        .hero-title {
            font-size: 1.85rem;
        }

        .stats-grid {
            grid-template-columns: 1fr;
        }

        .message-card {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


message_count = len(st.session_state.chat_history)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <span class="sidebar-label">Workspace</span>
            <p class="sidebar-copy">
                Use this workspace for Sharesource operations support, incident context,
                workflow guidance, and quick answers for day-to-day platform questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        f"""
        <div class="sidebar-card">
            <span class="sidebar-label">Conversation</span>
            <p class="sidebar-copy">Messages in session: <strong>{message_count}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-topbar">
            <svg class="brand-logo" viewBox="0 0 210 80" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Vantive logo">
                <defs>
                    <linearGradient id="vantiveBg" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#7b2677" />
                        <stop offset="55%" stop-color="#8d2e80" />
                        <stop offset="100%" stop-color="#6b1f70" />
                    </linearGradient>
                </defs>
                <rect width="210" height="80" rx="16" fill="url(#vantiveBg)" />
                <path d="M22 22h15l10 17 10-17h14L53 58H40L22 22z" fill="#fffaf7" opacity="0.98" />
                <text x="80" y="49" fill="#fffaf7" font-size="28" font-weight="800"
                    font-family="Avenir Next, Nunito Sans, Trebuchet MS, sans-serif"
                    letter-spacing="-0.6">Vantive</text>
            </svg>
            <span class="eyebrow">AWS Knowledge Assistant</span>
        </div>
        <h1 class="hero-title">Sharesource Ops Support</h1>
        <p class="hero-subtitle">
            A focused support workspace for operational guidance, troubleshooting,
            and fast access to context your team needs during daily execution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_col_1, stat_col_2 = st.columns(2)
with stat_col_1:
    st.markdown(
        f"""
        <div class="stat-card">
            <span class="stat-label">Session Activity</span>
            <div class="stat-value">{message_count}</div>
            <p class="stat-note">Messages captured in the current working session.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with stat_col_2:
    st.markdown(
        """
        <div class="stat-card">
            <span class="stat-label">Assistant Status</span>
            <div class="stat-value">Ready</div>
            <p class="stat-note">Retrieval QA is initialized and waiting for your next prompt.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="chat-shell">
        <div class="section-head">
            <div>
                <h2 class="section-title">Conversation</h2>
                <p class="section-copy">Responses are generated from the retrieval QA workflow.</p>
            </div>
            <div class="section-tag">Professional Workspace</div>
        </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        render_message(message["sender"], message["content"])
else:
    st.markdown(
        """
        <div class="empty-state">
            <h3 class="empty-title">Start with a sharper question</h3>
            <p class="empty-copy">
                Try asking about issue triage, operational procedures, escalation paths,
                service behavior, or the next action needed to resolve a support case.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="input-shell">', unsafe_allow_html=True)
with st.form("chat_form", clear_on_submit=True):
    input_col, button_col = st.columns([5.4, 1.2])

    with input_col:
        query = st.text_input(
            "Ask a question",
            key="user_query",
            placeholder="Ask about production issues differences in use cases, costs, hosting, security, or deployment...",
            label_visibility="collapsed",
        )

    with button_col:
        submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and query:
        st.session_state.chat_history.append({"sender": "User", "content": query})
        answer = get_answer(query, st.session_state.qa, st.session_state.memory)
        st.session_state.chat_history.append({"sender": "Bot", "content": answer})
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
