import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from chains import Chain
from utils import clean_and_truncate_text


def create_streamlit_app(llm, clean_and_truncate_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")

    st.title("📧 Personalized Cold Email Generator")

    col1, col2 = st.columns([2, 1])

    with col1:
        url_input = st.text_input(
            "Enter Job Posting URL:",
            value="https://www.linkedin.com/jobs/view/...",
            help="Paste the complete URL of the job posting"
        )

    with col2:
        st.markdown("### Profile: Vavilapalli Ravi Teja")
        st.markdown("- 🎓 IIT Dhanbad - Mathematics and Computing")
        st.markdown("- 💼 AI SDE Intern @ Intel")

    submit_button = st.button("Generate Email", type="primary")

    if submit_button:
        try:
            with st.spinner("Analyzing job posting..."):
                loader = WebBaseLoader([url_input])
                data = clean_and_truncate_text(loader.load().pop().page_content)
                jobs = llm.extract_jobs(data)

            with st.spinner("Crafting personalized email..."):
                for job in jobs:
                    st.success("Email Generated Successfully!")
                    st.markdown("### Generated Email")
                    email = llm.write_mail(job)
                    st.code(email, language='markdown')

                    # Add copy button
                    if st.button("📋 Copy Email"):
                        st.write("Email copied to clipboard!")

        except Exception as e:
            st.error(f"An Error Occurred: {e}")
            st.markdown("Please check if the URL is valid and accessible.")


if __name__ == "__main__":
    chain = Chain()
    create_streamlit_app(chain, clean_and_truncate_text)