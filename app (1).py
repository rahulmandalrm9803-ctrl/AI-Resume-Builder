import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import spacy  # removed for cloud deployment
import re
import time

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 900;
        color: #0D9488;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #0F172A;
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #1E293B;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #14B8A6;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 0.3rem;
    }
    .skill-match {
        background: #CCFBF1;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        color: #0D9488;
        margin: 3px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .skill-miss {
        background: #FEE2E2;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        color: #EF4444;
        margin: 3px;
        display: inline-block;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .resume-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 2rem;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #1E293B;
    }
    .course-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #0D9488;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .course-title { font-weight: 700; color: #0F172A; font-size: 0.95rem; }
    .course-platform { color: #0D9488; font-size: 0.82rem; margin-top: 2px; }
    .ats-high   { color: #10B981; font-weight: 800; }
    .ats-medium { color: #F59E0B; font-weight: 800; }
    .ats-low    { color: #EF4444; font-weight: 800; }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #0D9488;
    }
    .tip-box {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #166534;
        font-size: 0.88rem;
        margin-top: 1rem;
    }
    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Skills Database ───────────────────────────────────────────
SKILLS_DB = [
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "go",
    "sql", "html", "css", "bash", "scala", "kotlin", "swift",
    "react", "angular", "vue", "nodejs", "django", "flask", "fastapi",
    "streamlit", "spring boot", "express",
    "machine learning", "deep learning", "nlp", "computer vision",
    "data science", "statistics", "data analysis", "feature engineering",
    "tensorflow", "pytorch", "keras", "scikit-learn", "huggingface",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "git", "linux", "ci/cd", "jenkins", "github actions",
    "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
    "llm", "transformers", "langchain", "openai", "generative ai",
    "prompt engineering", "rag", "vector database", "pinecone",
    "agile", "scrum", "rest api", "graphql", "microservices",
    "power bi", "tableau", "excel", "spark", "hadoop", "airflow"
]

# ── Course Database ───────────────────────────────────────────
COURSE_DB = {
    "machine learning": [
        ("Machine Learning Specialization", "Coursera — Andrew Ng", "https://coursera.org/specializations/machine-learning-introduction"),
        ("ML A-Z: Hands-On Python", "Udemy", "https://udemy.com/course/machinelearning/"),
        ("ML Crash Course", "Google — FREE", "https://developers.google.com/machine-learning/crash-course"),
    ],
    "deep learning": [
        ("Deep Learning Specialization", "Coursera — deeplearning.ai", "https://coursera.org/specializations/deep-learning"),
        ("PyTorch for Deep Learning", "Udemy", "https://udemy.com/course/pytorch-for-deep-learning/"),
        ("Practical Deep Learning", "fast.ai — FREE", "https://fast.ai"),
    ],
    "python": [
        ("Complete Python Bootcamp", "Udemy — Jose Portilla", "https://udemy.com/course/complete-python-bootcamp/"),
        ("Python for Everybody", "Coursera", "https://coursera.org/specializations/python"),
        ("Python Official Tutorial", "Python Docs — FREE", "https://docs.python.org/3/tutorial/"),
    ],
    "nlp": [
        ("NLP Specialization", "Coursera — deeplearning.ai", "https://coursera.org/specializations/natural-language-processing"),
        ("HuggingFace NLP Course", "HuggingFace — FREE", "https://huggingface.co/learn/nlp-course/"),
        ("NLP with Python", "Udemy", "https://udemy.com/course/nlp-natural-language-processing-with-python/"),
    ],
    "llm": [
        ("LLM Bootcamp", "Full Stack Deep Learning — FREE", "https://fullstackdeeplearning.com/llm-bootcamp/"),
        ("LangChain for LLM Applications", "Udemy", "https://udemy.com/course/langchain/"),
        ("Prompt Engineering Guide", "DAIR.AI — FREE", "https://promptingguide.ai"),
    ],
    "generative ai": [
        ("Generative AI with LLMs", "Coursera — deeplearning.ai", "https://coursera.org/learn/generative-ai-with-llms"),
        ("ChatGPT Prompt Engineering", "DeepLearning.AI — FREE", "https://deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/"),
        ("Generative AI Fundamentals", "Google — FREE", "https://cloudskillsboost.google/paths/118"),
    ],
    "aws": [
        ("AWS Certified Cloud Practitioner", "Udemy — Stephane Maarek", "https://udemy.com/course/aws-certified-cloud-practitioner-new/"),
        ("AWS Training & Certification", "Amazon — FREE", "https://aws.amazon.com/training/"),
        ("Cloud Computing Basics", "Coursera", "https://coursera.org/learn/cloud-computing-basics"),
    ],
    "docker": [
        ("Docker & Kubernetes: Complete Guide", "Udemy", "https://udemy.com/course/docker-and-kubernetes-the-complete-guide/"),
        ("Docker Official Get Started", "Docker Docs — FREE", "https://docs.docker.com/get-started/"),
        ("DevOps Bootcamp", "Udemy — TechWorld with Nana", "https://udemy.com/course/devops-bootcamp/"),
    ],
    "sql": [
        ("Complete SQL Bootcamp", "Udemy — Jose Portilla", "https://udemy.com/course/the-complete-sql-bootcamp/"),
        ("SQL for Data Science", "Coursera", "https://coursera.org/learn/sql-for-data-science"),
        ("W3Schools SQL Tutorial", "W3Schools — FREE", "https://w3schools.com/sql/"),
    ],
    "tensorflow": [
        ("TensorFlow Developer Certificate", "Coursera — deeplearning.ai", "https://coursera.org/professional-certificates/tensorflow-in-practice"),
        ("TensorFlow Tutorials", "TensorFlow Docs — FREE", "https://tensorflow.org/tutorials"),
        ("Deep Learning with TensorFlow", "Udemy", "https://udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python/"),
    ],
    "pytorch": [
        ("PyTorch Complete Bootcamp", "Udemy", "https://udemy.com/course/pytorch-for-deep-learning/"),
        ("PyTorch Official Tutorials", "PyTorch — FREE", "https://pytorch.org/tutorials/"),
        ("Deep Learning with PyTorch", "fast.ai — FREE", "https://fast.ai"),
    ],
    "react": [
        ("Complete React Developer", "Udemy — Andrei Neagoie", "https://udemy.com/course/complete-react-developer-zero-to-mastery/"),
        ("React Official Docs", "React — FREE", "https://react.dev/learn"),
        ("Full Stack Open (React)", "University of Helsinki — FREE", "https://fullstackopen.com/"),
    ],
    "data science": [
        ("Data Science Specialization", "Coursera — Johns Hopkins", "https://coursera.org/specializations/jhu-data-science"),
        ("Data Science Bootcamp", "Udemy — 365 Data Science", "https://udemy.com/course/the-data-science-course-complete-data-science-bootcamp/"),
        ("Kaggle Learn", "Kaggle — FREE", "https://kaggle.com/learn"),
    ],
}
DEFAULT_COURSES = [
    ("AI For Everyone", "Coursera — Andrew Ng — FREE", "https://coursera.org/learn/ai-for-everyone"),
    ("Google AI Essentials", "Google — FREE", "https://grow.google/certificates/ai-essentials/"),
    ("Elements of AI", "University of Helsinki — FREE", "https://elementsofai.com"),
]

# ── Helper Functions ──────────────────────────────────────────
def extract_skills(text):
    text_lower = text.lower()
    return list(set([skill for skill in SKILLS_DB if skill in text_lower]))

def calculate_ats_score(resume_text, job_description):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([resume_text, job_description])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return min(round(score * 100 * 1.4, 1), 99.0)
    except:
        return 0.0

def get_skill_gap(user_skills, required_skills):
    user_lower = [s.lower().strip() for s in user_skills]
    matched, missing = [], []
    for skill in required_skills:
        if any(skill in u or u in skill for u in user_lower):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing

def generate_resume(name, email, phone, linkedin, job_role,
                    skills, experience, education, achievements):
    skills_str = " • ".join(skills)
    top3 = ", ".join(skills[:3]) if len(skills) >= 3 else ", ".join(skills)
    summary = (
        f"Results-driven {job_role} with strong expertise in {top3}. "
        f"Passionate about leveraging cutting-edge technologies to solve real-world "
        f"problems and deliver high-impact solutions. Proven ability to collaborate "
        f"in fast-paced, agile environments and drive innovation from concept to deployment."
    )
    linkedin_line = f"LinkedIn : {linkedin}" if linkedin else ""
    ach_section = ""
    if achievements.strip():
        ach_section = f"""
ACHIEVEMENTS & CERTIFICATIONS
{'─'*55}
{achievements}
"""
    resume = f"""{'━'*60}
{name.upper()}
{job_role}
{'━'*60}
Email    : {email}
Phone    : {phone}
{linkedin_line}
{'─'*60}

PROFESSIONAL SUMMARY
{'─'*55}
{summary}

TECHNICAL SKILLS
{'─'*55}
{skills_str}

WORK EXPERIENCE
{'─'*55}
{experience}

EDUCATION
{'─'*55}
{education}
{ach_section}
{'━'*60}
Generated by AI Resume Builder | Generative AI Project
GitHub: github.com/rahulmandalrm9803-ctrl/AI-Resume-Builder
{'━'*60}"""
    return resume

def recommend_courses(missing_skills):
    recommendations = {}
    for skill in missing_skills:
        skill_lower = skill.lower()
        courses = None
        for key in COURSE_DB:
            if key in skill_lower or skill_lower in key:
                courses = COURSE_DB[key]
                break
        recommendations[skill] = courses if courses else DEFAULT_COURSES
    return recommendations

def ats_color_class(score):
    if score >= 70: return "ats-high", "🟢 Excellent Match!"
    elif score >= 50: return "ats-medium", "🟡 Moderate Match"
    else: return "ats-low", "🔴 Needs Improvement"

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("## 🤖 AI Resume Builder")
    st.markdown("---")
    st.markdown("### 📌 About")
    st.info(
        "This app uses **Generative AI + NLP** to:\n\n"
        "✅ Build tailored resumes\n\n"
        "✅ Analyze skill gaps\n\n"
        "✅ Score ATS compatibility\n\n"
        "✅ Recommend courses"
    )
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.code("Python • Streamlit\nspaCy • scikit-learn\nPlotly • NLTK\nOpenAI (optional)", language="text")
    st.markdown("---")
    st.markdown(
        "**GitHub:**\n[rahulmandalrm9803-ctrl](https://github.com/rahulmandalrm9803-ctrl/AI-Resume-Builder)",
        unsafe_allow_html=True
    )
    st.caption("BTech Group Project | Generative AI Course")

# ── Main Header ───────────────────────────────────────────────
st.markdown("<div class='main-header'>🤖 AI-Powered Resume Builder</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>& Skill Recommendation System — Powered by Generative AI + NLP</div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Resume Generator",
    "📊 Skill Gap Analyzer",
    "🎓 Course Recommender",
    "ℹ️ How It Works"
])

# ═══════════════════════════════════════════════════
# TAB 1 — RESUME GENERATOR
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>📄 Build Your AI-Powered Resume</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Personal Information**")
        name     = st.text_input("Full Name *", placeholder="e.g. Rahul Sharma")
        email    = st.text_input("Email *", placeholder="e.g. rahul@email.com")
        phone    = st.text_input("Phone *", placeholder="e.g. +91-9876543210")
        linkedin = st.text_input("LinkedIn URL (optional)", placeholder="linkedin.com/in/yourname")
        job_role = st.text_input("Target Job Role *", placeholder="e.g. Machine Learning Engineer")

    with col2:
        st.markdown("**Skills & Background**")
        skills_input = st.text_area(
            "Your Skills * (comma separated)",
            placeholder="e.g. Python, Machine Learning, TensorFlow, SQL, NLP, Docker",
            height=90
        )
        experience = st.text_area(
            "Work Experience *",
            placeholder="e.g.\nML Engineer Intern @ TechCorp (2023–24)\n- Built churn prediction model with 85% accuracy\n- Deployed Flask REST API for model serving",
            height=110
        )
        education = st.text_area(
            "Education *",
            placeholder="e.g. B.Tech Computer Science | XYZ University | 2024 | CGPA: 8.5/10",
            height=60
        )
        achievements = st.text_area(
            "Achievements & Certifications (optional)",
            placeholder="e.g.\n- AWS Cloud Practitioner Certified\n- Winner — Smart India Hackathon 2023",
            height=70
        )

    st.markdown("")
    generate_btn = st.button("🚀 Generate AI Resume", type="primary", use_container_width=True)

    if generate_btn:
        if not all([name, email, phone, job_role, skills_input, experience, education]):
            st.error("⚠️ Please fill all required (*) fields!")
        else:
            skills = [s.strip() for s in skills_input.split(",") if s.strip()]
            with st.spinner("🤖 AI is crafting your resume..."):
                time.sleep(1.5)
                resume_text = generate_resume(
                    name, email, phone, linkedin, job_role,
                    skills, experience, education, achievements
                )

            st.success("✅ Resume Generated Successfully!")
            st.markdown("")

            # ATS Score against job role keywords
            jd_mock = f"{job_role} {skills_input} python machine learning data"
            score = calculate_ats_score(resume_text, jd_mock)
            css_class, label = ats_color_class(score)

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><div class='metric-value'>{score}</div><div class='metric-label'>ATS Score / 100</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-value'>{len(skills)}</div><div class='metric-label'>Skills Listed</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-value {css_class}'>{label}</div><div class='metric-label'>ATS Status</div></div>", unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**📋 Your AI-Generated Resume:**")
            st.markdown(f"<div class='resume-box'>{resume_text}</div>", unsafe_allow_html=True)

            st.download_button(
                label="📥 Download Resume as .txt",
                data=resume_text,
                file_name=f"{name.replace(' ', '_')}_AI_Resume.txt",
                mime="text/plain",
                use_container_width=True
            )

            st.markdown("<div class='tip-box'>💡 <b>Tip:</b> Paste your target Job Description in the <b>Skill Gap Analyzer</b> tab to see exactly which skills to add to improve your ATS score!</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 2 — SKILL GAP ANALYZER
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>📊 Skill Gap & ATS Analyzer</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        user_skills_input = st.text_area(
            "Your Current Skills *",
            placeholder="e.g. Python, machine learning, SQL, pandas, Flask",
            height=120
        )
    with col2:
        jd_input = st.text_area(
            "Paste Job Description *",
            placeholder="Copy and paste the full job description here from LinkedIn, Naukri, etc.",
            height=120
        )

    analyze_btn = st.button("🔍 Analyze My Profile", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_skills_input or not jd_input:
            st.error("⚠️ Please fill both fields!")
        else:
            user_skills  = [s.strip() for s in user_skills_input.split(",") if s.strip()]
            jd_skills    = extract_skills(jd_input)
            matched, missing = get_skill_gap(user_skills, jd_skills)
            ats          = calculate_ats_score(user_skills_input, jd_input)
            css_class, label = ats_color_class(ats)

            with st.spinner("🔍 Analyzing your profile against the job description..."):
                time.sleep(1)

            st.markdown("---")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><div class='metric-value {css_class}'>{ats}</div><div class='metric-label'>ATS Score / 100</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#10B981'>{len(matched)}</div><div class='metric-label'>Skills Matched ✅</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#EF4444'>{len(missing)}</div><div class='metric-label'>Skills Missing ❌</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><div class='metric-value'>{len(jd_skills)}</div><div class='metric-label'>Total JD Skills</div></div>", unsafe_allow_html=True)

            st.markdown("")
            st.markdown(f"**ATS Status: <span class='{css_class}'>{label}</span>**", unsafe_allow_html=True)

            # Skill chips
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**✅ Skills You Have (Matched):**")
                if matched:
                    st.markdown(" ".join([f"<span class='skill-match'>{s}</span>" for s in matched]), unsafe_allow_html=True)
                else:
                    st.warning("No direct skill matches found.")

            with col_b:
                st.markdown("**❌ Skills You're Missing:**")
                if missing:
                    st.markdown(" ".join([f"<span class='skill-miss'>{s}</span>" for s in missing]), unsafe_allow_html=True)
                else:
                    st.success("🎉 You match all required skills!")

            # Bar Chart
            st.markdown("")
            st.markdown("**📊 Visual Skill Gap Chart:**")
            all_skills = matched + missing
            if all_skills:
                colors = ["#0D9488"] * len(matched) + ["#EF4444"] * len(missing)
                status = ["✅ Matched"] * len(matched) + ["❌ Missing"] * len(missing)
                fig = go.Figure(go.Bar(
                    x=[1] * len(all_skills),
                    y=all_skills,
                    orientation="h",
                    marker_color=colors,
                    text=status,
                    textposition="inside",
                    insidetextfont=dict(color="white", size=12)
                ))
                fig.update_layout(
                    title="Your Skills vs Job Requirements",
                    xaxis_visible=False,
                    height=max(300, len(all_skills) * 38),
                    plot_bgcolor="#F8FAFC",
                    paper_bgcolor="white",
                    margin=dict(l=10, r=10, t=40, b=10),
                    font=dict(family="Inter, sans-serif")
                )
                st.plotly_chart(fig, use_container_width=True)

            if missing:
                st.markdown("<div class='tip-box'>💡 <b>Next Step:</b> Go to the <b>Course Recommender</b> tab and paste your missing skills to get top course suggestions!</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
# TAB 3 — COURSE RECOMMENDER
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>🎓 Personalized Course Recommendations</div>", unsafe_allow_html=True)

    st.markdown("Enter the skills you want to learn or are missing from a job description:")
    gap_input = st.text_area(
        "Skill Gaps (comma separated)",
        placeholder="e.g. deep learning, aws, docker, nlp, generative ai",
        height=80
    )

    rec_btn = st.button("🎓 Get Course Recommendations", type="primary", use_container_width=True)

    if rec_btn:
        if not gap_input.strip():
            st.error("⚠️ Please enter at least one skill!")
        else:
            gaps = [s.strip() for s in gap_input.split(",") if s.strip()]
            with st.spinner("🔍 Finding best courses for you..."):
                time.sleep(0.8)
                recs = recommend_courses(gaps)

            st.success(f"✅ Found courses for {len(gaps)} skill(s)!")
            st.markdown("")

            for skill, courses in recs.items():
                st.markdown(f"### 📚 {skill.title()}")
                for i, (cname, platform, url) in enumerate(courses[:3], 1):
                    st.markdown(
                        f"<div class='course-card'>"
                        f"<div class='course-title'>{i}. {cname}</div>"
                        f"<div class='course-platform'>🏛️ {platform}</div>"
                        f"<div style='margin-top:6px'><a href='{url}' target='_blank'>🔗 Open Course →</a></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                st.markdown("")

# ═══════════════════════════════════════════════════
# TAB 4 — HOW IT WORKS
# ═══════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>ℹ️ How This System Works</div>", unsafe_allow_html=True)

    steps = [
        ("1️⃣", "Enter Your Profile", "Fill in your name, skills, experience, education, and target job role in the Resume Generator tab."),
        ("2️⃣", "AI Generates Resume", "Our NLP engine crafts a professional, ATS-optimized resume tailored to your target role — in seconds."),
        ("3️⃣", "Paste Job Description", "Go to Skill Gap Analyzer and paste any job description from LinkedIn, Naukri, Indeed, etc."),
        ("4️⃣", "Get Skill Gap Report", "The system uses TF-IDF + Cosine Similarity to compare your skills with the job requirements and shows matched vs missing skills."),
        ("5️⃣", "Get Course Suggestions", "The Course Recommender maps your skill gaps to the best online courses from Coursera, Udemy, and free platforms."),
    ]

    for emoji, title, desc in steps:
        col1, col2 = st.columns([0.08, 0.92])
        with col1:
            st.markdown(f"<div style='font-size:2rem;text-align:center'>{emoji}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{title}**")
            st.markdown(desc)
        st.markdown("")

    st.markdown("---")
    st.markdown("### 🧠 AI Techniques Used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| Technique | Purpose |
|---|---|
| **TF-IDF Vectorization** | ATS Score Calculation |
| **Cosine Similarity** | Skill Matching |
| **NLP (spaCy)** | Skill Extraction from JD |
| **Content Filtering** | Course Recommendation |
        """)
    with col2:
        st.markdown("""
| Library | Role |
|---|---|
| **Streamlit** | GUI Framework |
| **scikit-learn** | TF-IDF & Similarity |
| **Plotly** | Interactive Charts |
| **spaCy / NLTK** | Text Processing |
        """)

    st.markdown("---")
    st.markdown("**🔗 GitHub Repository:** [github.com/rahulmandalrm9803-ctrl/AI-Resume-Builder](https://github.com/rahulmandalrm9803-ctrl/AI-Resume-Builder)")
    st.caption("BTech Group Project | Generative AI Course | Python + Streamlit + NLP")
