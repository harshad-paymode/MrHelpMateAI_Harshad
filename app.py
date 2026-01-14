import streamlit as st
from src.core.logging_config import logger
from src.prompts import get_prompt_template
from src.core.models import get_generator
from src.moderation import moderation_check
from src.routing import get_router_message
from src.pipeline import execute_chain
from src.core.config import MODELS, PATHS
from src.evaluations import get_evaluation_results
import os

os.environ['MISTRAL_API_KEY'] = MODELS.MODEL_API_KEY
st.title("MR HELPMATE")

# Custom CSS to style the expander
st.markdown("""
    <style>
    /* Style the header text color to blue */
    .streamlit-expanderHeader p {
        color: #2e9aff;
        font-weight: bold;
    }
    
    /* Style the content background to black and text to white for readability */
    div[data-testid="stExpander"] div[role="region"] {
        background-color: black;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.expander("About MrHelpMate Assistant", expanded=True):
    st.markdown("""
    This assistant helps you with:

    - MrHelpMate helps you understand insurance policy documents — clearly and with citations.
    - Answer Quality: How well the answer actually addresses your question.
    - Grounded Accuracy: How much of the answer is directly backed by the policy documents.
    - Ask questions about coverage, eligibility, exclusions, limits, waiting periods, claims, and processes.
    - Answers are based only on the policy documents — no guessing.
    - Avoid small-talk, medical/legal advice, personal opinion, or questions unrelated to the policy.
    - If something isn’t in the documents, you’ll be told — rather than receiving a made-up answer.
    """)

query = st.chat_input("Please ask your query about the policy Document")

# Initialize chat history
if "messages" not in st.session_state:
    logger.info("Chat history initialized")
    st.session_state.messages = []

# Sidebar button to start a new chat
if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.rerun()

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message['flag'] == 'rag':
            st.write("\n\nFeel free to download and read the policy pdf")
            with open(PATHS.POLICY_PDF_PATH, "rb") as file:
                st.download_button(
                    label="Policy PDF",
                    data=file,
                    file_name="insurance_policy.pdf",
                    mime="application/pdf",
                    icon=":material/picture_as_pdf:",
                    key=f"download_btn_{i}"  # Added unique key using the index
                )

if query:
    logger.info("User Query is retrieved")
    # Display user message and add to history
    st.session_state.messages.append({"role": "user", "content": query, "flag": "rag"})
    with st.chat_message("user"):
        st.write(query)
    
    # Display "Searching..." while processing
    with st.status("Searching..."):
        prompt = get_prompt_template()
        moderator, block_chain = moderation_check()
        llm_gen = get_generator()
        small_talk = get_router_message()

        logger.info("Variables Retrieved, Calling execute chain")
        response = execute_chain(query, prompt, llm_gen, moderator, block_chain, small_talk)
    
    with st.chat_message("assistant"):
        content = response["response"]["answer"].content
        context = response["response"]["context"]

        faithfullness_score, answer_relevancy_score = get_evaluation_results(
            query, content, context
        )

        # Convert to percentage (assuming scores are 0–1)
        faith_pct = round(faithfullness_score * 100)
        rel_pct = round(answer_relevancy_score * 100)

        # Sidebar big cards for scores
        with st.sidebar:
            st.markdown(
                """
                <style>
                .score-card {
                    padding: 12px 16px;
                    margin-top: 8px;
                    margin-bottom: 12px;
                    border-radius: 8px;
                    background-color: #0e1117;
                    color: white;
                    border: 1px solid #2e9aff;
                }
                .score-title {
                    font-size: 18px;
                    font-weight: 700;
                    margin-bottom: 4px;
                }
                .score-value {
                    font-size: 32px;
                    font-weight: 800;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-title">Answer Quality</div>
                    <div class="score-value">{rel_pct}%</div>
                </div>
                <div class="score-card">
                    <div class="score-title">Grounded Accuracy</div>
                    <div class="score-value">{faith_pct}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write(content)

        # Display the PDF download button immediately after the text
        if response["flag"] == 'rag':
            st.write("\n\nFeel free to download and read the policy pdf")
            with open(PATHS.POLICY_PDF_PATH, "rb") as file:
                st.download_button(
                    label="Policy PDF",
                    data=file,
                    file_name="insurance_policy.pdf",
                    mime="application/pdf",
                    icon=":material/picture_as_pdf:"
                )
                logger.info("PDF attached below the chat")

    st.session_state.messages.append(
        {"role": "assistant", "content": content, "flag": response['flag']}
    )