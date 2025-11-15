import streamlit as st
import requests
import time

FASTAPI_URL = "http://localhost:8000"

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Agentic RAG Chat", layout="wide")

# --------------------------
# Session State Initialization
# --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "options" not in st.session_state:
    st.session_state.options = {"model": "default", "temperature": 0.3}

if "last_input_sent" not in st.session_state:
    st.session_state.last_input_sent = ""

# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.header("⚙️ LLM Options")
    st.session_state.options["model"] = st.selectbox(
        "Model",
        ["Groq", "OpenAI", "Gemini"],
        index=0,
    )

    st.markdown("---")
    st.header("📄 Document Upload")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
        try:
            res = requests.post(f"{FASTAPI_URL}/upload_pdf", files=files, timeout=10)
            res.raise_for_status()
            st.success("PDF uploaded & processed successfully!")
            st.session_state.pdf_uploaded = True
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend. ({e})")

    if st.session_state.pdf_uploaded:
        if st.button("❌ Clear PDF"):
            st.session_state.pdf_uploaded = False
            st.success("PDF cleared. You can re-upload another.")

# --------------------------
# Title
# --------------------------
st.title("🤖 AI Study & Research Copilot")
st.write("Chat with Everything You Learn.")

# --------------------------
# Chat Display Function
# --------------------------
def display_chat():
    for role, content in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"""
                <div style='text-align:right; margin:8px;'>
                    <div style='display:inline-block; padding:10px 14px;
                        background:#3b82f6; color:white; border-radius:12px;
                        max-width:70%; word-wrap:break-word;'>{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='text-align:left; margin:8px;'>
                    <div style='display:inline-block; padding:10px 14px;
                        background:white; color:black; border-radius:12px;
                        border:1px solid #eee; max-width:70%; word-wrap:break-word;'>{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Auto-scroll
    st.markdown("""
    <script>
        var chatBox = window.parent.document.querySelector('.main');
        if (chatBox) { chatBox.scrollTop = chatBox.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)

# Display existing chat
display_chat()

# --------------------------
# Chat Input (Pinned at Bottom)
# --------------------------
if user_input := st.chat_input("Type your message here..."):
    # Only process new input
    if user_input != st.session_state.last_input_sent:
        st.session_state.last_input_sent = user_input

        # Add user message
        st.session_state.chat_history.append(("user", user_input))
        display_chat()  # Show user message immediately

        # Determine if input is a URL for ingestion
        is_link = user_input.startswith("http://") or user_input.startswith("https://")

        with st.spinner("Thinking..."):
            try:
                if is_link:
                    # Send to ingestion endpoint
                    res = requests.post(
                        f"{FASTAPI_URL}/chat_or_ingest",
                        data={"message": user_input},
                        timeout=60
                    )
                    res.raise_for_status()
                    data = res.json()

                    if data["type"] == "ingestion":
                        answer = "\n".join([
                            f"{r.get('message','Ingested')} ({r.get('ingested_chunks',0)} chunks)"
                            for r in data["results"]
                        ])
                    else:
                        answer = data.get("answer", "❌ No answer returned.")
                else:
                    # Normal chat query
                    res = requests.get(
                        f"{FASTAPI_URL}/chat",
                        params={"q": user_input},
                        timeout=30
                    )
                    res.raise_for_status()
                    data = res.json()
                    answer = data.get("answer", "❌ No answer returned.")
            except requests.exceptions.RequestException as e:
                answer = f"❌ Cannot reach backend. ({e})"
            except ValueError:
                answer = "❌ Invalid JSON response from backend."

        # Add bot placeholder (empty message)
        st.session_state.chat_history.append(("assistant", ""))
        bot_index = len(st.session_state.chat_history) - 1

        # Typing animation in placeholder
        built = ""
        placeholder = st.empty()
        for ch in answer:
            built += ch
            placeholder.markdown(
                f"""
                <div style='text-align:left; margin:8px;'>
                    <div style='display:inline-block; padding:10px 14px;
                        background:white; color:black; border-radius:12px;
                        border:1px solid #eee; max-width:70%; word-wrap:break-word;'>{built}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Replace placeholder with final answer in chat history
        st.session_state.chat_history[bot_index] = ("assistant", answer)
