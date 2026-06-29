import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
import re

st.set_page_config(page_title="StudyGPT", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #dbeafe; }
html, body, [class*="css"] { color: black; }
section[data-testid="stSidebar"] { background-color: #bfdbfe; }
h1,h2,h3,h4,h5,h6,p,div,label { color: black !important; }
.stTextInput > div > div > input {
    background-color: white; color: black;
    border-radius: 12px; border: 2px solid #93c5fd; padding: 12px;
}
.stButton > button {
    background-color: white; color: black;
    border-radius: 12px; border: none;
    padding: 10px 18px; font-weight: bold;
}
[data-testid="stFileUploader"] section {
    background-color: white; border: 2px dashed #60a5fa;
    border-radius: 15px; padding: 10px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("StudyGPT")
    st.write("AI Study Assistant")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

st.title("Ask anything about your PDF")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    text = "".join(p.extract_text() or "" for p in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["AI Tutor", "Smart Notes", "Flashcards", "Quiz", "Important Questions"])

    # AI TUTOR
    with tab1:
        question = st.text_input("Ask anything from the PDF")
        if question:
            qv = vectorizer.transform([question])
            best = cosine_similarity(qv, vectors).flatten().argmax()
            context = chunks[best]
            prompt = f"""You are an intelligent study tutor. Answer the student's question in detail. Explain concepts clearly. Use simple language. Make the answer educational.

Context: {context}

Question: {question}"""
            with st.spinner("Thinking..."):
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            answer = response["message"]["content"]
            st.markdown(f"""
            <div style="background-color:#93c5fd;padding:20px;border-radius:15px;margin-bottom:20px;">
            <h3>You</h3><p>{question}</p>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color:white;padding:25px;border-radius:15px;border:2px solid #93c5fd;">
            <h3>StudyGPT</h3>
            <p style="font-size:18px;line-height:1.8;">{answer}</p>
            </div>""", unsafe_allow_html=True)

    # SMART NOTES
    with tab2:
        if st.button("Generate Notes"):
            prompt = f"""Create clean study notes from this PDF.
RULES: Focus on concepts, avoid citations and author names, use bullet points, make it concise but informative.

PDF: {text[:5000]}"""
            with st.spinner("Generating notes..."):
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            st.markdown(f"""
            <div style="background-color:white !important;padding:30px;border-radius:20px;border:2px solid #93c5fd;font-size:18px;line-height:1.8;">
            {response["message"]["content"]}
            </div>""", unsafe_allow_html=True)

    # FLASHCARDS
    with tab3:
        if "flashcards" not in st.session_state:
            st.session_state.flashcards = None
        if st.button("Generate Flashcards"):
            prompt = f"""Generate 8 smart flashcards from this PDF.
RULES: Focus on concepts, avoid author names and references, questions should sound natural, answers should help in studying.

FORMAT:
Q: question
A: answer

PDF: {text[:5000]}"""
            with st.spinner("Creating flashcards..."):
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            st.session_state.flashcards = response["message"]["content"]

        if st.session_state.flashcards:
            cards = [c for c in st.session_state.flashcards.split("Q:") if c.strip()]
            for card in cards:
                parts = card.split("A:")
                if len(parts) == 2:
                    question_text = parts[0].strip()
                    answer_text = parts[1].strip()
                    st.markdown(f"""
                    <div style="background-color:#0f172a;padding:30px;border-radius:20px;margin-bottom:25px;border:2px solid #2563eb;">
                    <h2 style="color:white !important;">Question</h2>
                    <p style="color:white !important;font-size:22px;line-height:1.7;">{question_text}</p>
                    <hr style="border:1px solid #3b82f6;margin:25px 0;">
                    <h2 style="color:white !important;">Answer</h2>
                    <p style="color:white !important;font-size:20px;line-height:1.8;">{answer_text}</p>
                    </div>""", unsafe_allow_html=True)

    # QUIZ
    with tab4:
        if "quiz_data" not in st.session_state:
            st.session_state.quiz_data = []
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = False

        if st.button("Generate Quiz"):
            st.session_state.quiz_data = []
            st.session_state.quiz_submitted = False
            prompt = f"""Create exactly 10 high-quality MCQ questions from this PDF.
RULES: Focus on concepts, avoid author names and citations, questions should feel like real exam questions, each MUST have exactly 4 options, mention correct answer clearly.

FORMAT EXACTLY LIKE THIS:
QUESTION: What is CNN used for?
A) Image classification
B) Cooking
C) Banking
D) Driving
ANSWER: A

QUESTION: next question...

PDF: {text[:6000]}"""
            with st.spinner("Generating 10 quiz questions..."):
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            quiz_text = response["message"]["content"]
            pattern = r"QUESTION:\s*(.*?)\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\nANSWER:\s*([ABCD])"
            matches = re.findall(pattern, quiz_text, re.DOTALL)
            st.session_state.quiz_data = [
                {"question": m[0].strip(),
                 "options": [f"A) {m[1].strip()}", f"B) {m[2].strip()}", f"C) {m[3].strip()}", f"D) {m[4].strip()}"],
                 "answer": m[5].strip()}
                for m in matches
            ]
            if len(st.session_state.quiz_data) < 5:
                st.warning("Could not parse enough questions. Try regenerating.")

        if st.session_state.quiz_data and not st.session_state.quiz_submitted:
            st.markdown(f"**Answer all {len(st.session_state.quiz_data)} questions, then click Submit.**")
            user_answers = {}
            for i, q in enumerate(st.session_state.quiz_data):
                st.markdown(f"**Q{i+1}. {q['question']}**")
                user_answers[i] = st.radio("Choose an option", q["options"], key=f"quiz_{i}", index=None)
                st.markdown("---")

            if st.button("Submit Quiz"):
                if any(v is None for v in user_answers.values()):
                    st.warning("Please answer all questions before submitting.")
                else:
                    st.session_state.quiz_answers = user_answers
                    st.session_state.quiz_submitted = True
                    st.rerun()

        if st.session_state.quiz_submitted and st.session_state.quiz_data:
            answers = st.session_state.get("quiz_answers", {})
            score = sum(1 for i, q in enumerate(st.session_state.quiz_data) if answers.get(i, "").startswith(q["answer"]))
            total = len(st.session_state.quiz_data)
            pct = int(score / total * 100)
            grade = "Excellent!" if pct >= 80 else "Good job!" if pct >= 60 else "Keep studying!"
            st.markdown(f"""
            <div style="background-color:white;padding:30px;border-radius:20px;border:2px solid #93c5fd;text-align:center;margin-bottom:20px;">
            <h1 style="color:black !important;font-size:52px;">{score}/{total}</h1>
            <p style="color:black !important;font-size:20px;">{pct}% &nbsp;·&nbsp; {grade}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("### Review")
            for i, q in enumerate(st.session_state.quiz_data):
                user_ans = answers.get(i, "")
                correct = user_ans.startswith(q["answer"])
                label = "Correct" if correct else "Wrong"
                correct_opt = next((o for o in q["options"] if o.startswith(q["answer"])), q["answer"])
                with st.expander(f"[{label}] Q{i+1}: {q['question'][:70]}"):
                    st.markdown(f"**Your answer:** {user_ans}")
                    if not correct:
                        st.markdown(f"**Correct answer:** {correct_opt}")

            if st.button("Retake Quiz"):
                st.session_state.quiz_data = []
                st.session_state.quiz_submitted = False
                st.rerun()

    # IMPORTANT QUESTIONS
    with tab5:
        if st.button("Generate Important Questions"):
            prompt = f"""Generate 8 important long-answer study questions from this PDF.
RULES: Conceptual questions only, avoid factual recall, avoid authors and citations, questions should help for exams and interviews, number each question.

PDF: {text[:5000]}"""
            with st.spinner("Generating questions..."):
                response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
            st.markdown(f"""
            <div style="background-color:white;padding:30px;border-radius:20px;border:2px solid #93c5fd;font-size:18px;line-height:2;">
            {response["message"]["content"]}
            </div>""", unsafe_allow_html=True)