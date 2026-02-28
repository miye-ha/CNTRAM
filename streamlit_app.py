import streamlit as st
from rag import RAGClient
import platform
import time
from datetime import datetime
import conf

# Initialize RAG client
rag_client = RAGClient()

# Page configuration
st.set_page_config(
    page_title="Nursing Record Coding",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styles
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        border-radius: 5px;
        padding: 10px;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
    .stProgress .st-bo {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Nursing Record Coding")


# Create two-column layout
col1, col2 = st.columns(2)

# Initialize state
if 'nursing_record' not in st.session_state:
    st.session_state['nursing_record'] = ''
if 'response_text' not in st.session_state:
    st.session_state['response_text'] = ''
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None

# Left column: Input area
with col1:
    st.markdown("### Input Area")
    # Use session_state to store nursing record
    st.session_state.nursing_record = st.text_area(
        "Please enter nursing record content",
        value=st.session_state.nursing_record,
        height=200,
        placeholder="Enter nursing record here...",
        key="nursing_record_input"
    )

    # Analyze button
    if st.button("Analyze Codes", type="primary", key="analyze_button", disabled=st.session_state.is_processing):
        if st.session_state.nursing_record.strip():
            try:
                st.session_state.is_processing = True
                st.session_state.response_text = ""
                
                # Show progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_text = st.empty()
                
                start_time = time.time()
                
                # Update status
                status_text.text("Analyzing...")
                progress_bar.progress(20)
                
                # Execute query
                filtered_response = rag_client.query(st.session_state.nursing_record)
                progress_bar.progress(60)
                
                # Update result
                st.session_state.response_text = filtered_response.strip()
                progress_bar.progress(100)
                
                # Update last update time
                st.session_state['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Show processing time
                elapsed_time = time.time() - start_time
                time_text.text(f"Processing time: {elapsed_time:.2f} seconds")
                
                # Clear progress display
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                time_text.empty()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.response_text = f"Analysis error: {str(e)}"
            finally:
                st.session_state.is_processing = False
        else:
            st.warning("Please enter nursing record content")
            st.session_state.response_text = "Please enter nursing record content"

# Right column: Result display area
with col2:
    st.markdown("### Analysis Result")
    

    # Result display
    result_area = st.text_area(
        "Analysis Result",
        value=st.session_state.response_text,
        height=200,
        disabled=True,
        key="result_display"
    )
    
    # Copy button
    if st.button("Copy Result", type="secondary", key="copy_button", disabled=not st.session_state.response_text):
        try:
            # Use JavaScript to copy text
            st.markdown(
                f"""
                <script>
                    navigator.clipboard.writeText(`{st.session_state.response_text}`);
                </script>
                """,
                unsafe_allow_html=True
            )
            st.success("Copied to clipboard!")
        except Exception as e:
            st.error(f"Copy failed: {str(e)}")
