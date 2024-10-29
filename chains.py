import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            groq_api_key="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",
            model_name="llama-3.1-70b-versatile"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            Extract detailed information from this job posting and return it in JSON format with these keys:
            - `role`: The exact job title
            - `experience`: Required years and type of experience
            - `skills`: List of technical and soft skills required
            - `description`: Detailed job responsibilities
            - `company`: Company name
            - `keywords`: Key technical terms and technologies mentioned
            - `focus_areas`: Main areas of work (e.g., ML, Data Analysis, AI Engineering)

            Focus only on roles related to Data Science, ML Engineering, AI Development, or Data Analysis.
            Return only the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Vavilapalli Ravi Teja, currently pursuing Integrated Master of Technology in Mathematics and Computing at IIT Dhanbad 
            with a strong background in AI and Data Science. Write a professional cold email considering these guidelines:

            1. CURRENT ROLE:
            - AI Software Development Engineer Intern at Intel Corporation
            - Working on AI workload enablement, PyTorch models, and performance analysis
            - Strong experience with VTune analysis and Intel Arc GPU
            - Hands-on experience with Llama-3.1, AlexNet, GPT models

            2. PREVIOUS EXPERIENCE:
            - Data Science Intern at Get Better Studios
            - Developed data collection solutions and performed analysis
            - Created visualizations using Power BI
            - Experience with web scraping and data pipeline development

            3. NOTABLE PROJECTS:
            - IPL WIN PREDICTOR: ML model with Streamlit deployment
            - ODI WC 2023 Analysis: Comprehensive Power BI dashboard
            - Absenteeism Model: Predictive modeling with Tableau visualization

            4. TECHNICAL EXPERTISE:
            - Programming: Python, C++, R
            - ML/AI: PyTorch, OpenVino, Deep Learning
            - Data Analysis: SQL, Tableau, Power BI, MS Excel
            - Core Concepts: DSA, Statistical Analysis, GPU Computing

            5. ACHIEVEMENTS:
            - Finalist in TRILYTICS'23 at IIM Calcutta
            - JEE Advanced 2020 Rank: 6537
            - 5 Star Rating in HackerRank Problem Solving

            Guidelines for email generation:
            1. Analyze the job keywords and match them with relevant experiences
            2. Highlight projects and internships that align with the role's requirements
            3. Demonstrate understanding of the company's needs
            4. Keep the tone professional but enthusiastic
            5. Make the email unique for each application
            6. Include specific achievements relevant to the role
            7. End with contact information:
               - Phone: +91-6303590620
               - Email: ravitejavavilapalli21@gmail.com
               - LinkedIn: linkedin.com/in/RaviTeja

            Write a personalized email that specifically addresses the job requirements while highlighting relevant experience.
            ### EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job)})
        return res.content