# AI Pre-Screening Interviewer (Cyber-Midnight)

A futuristic, AI-powered recruitment tool designed to automate the initial candidate screening process. This agent analyzes Job Descriptions (JDs) and CVs, generates targeted technical questions, and produces comprehensive evaluation reports using LLMs and Vector Search.

![UI Screenshot](https://img.shields.io/badge/UI-Cyber--Midnight-cyan)
![Tech](https://img.shields.io/badge/Powered%20By-Groq%20%7C%20LLaMA%203.1-blueviolet)

## 📖 What is this Project?
The **AI Pre-Screening Interviewer** is an intelligent assistant designed to bridge the gap between initial job applications and technical interviews. In modern recruitment, hiring managers often spend hours manually screening CVs against complex job descriptions. This project automates that "first look," acting as a specialized technical recruiter that can conduct a live preliminary interview.

## 🤖 What does it do?
The system performs a multi-stage analysis to ensure high-quality screening:

1. **Document Intelligence**: It reads and "understands" both the Job Description and the Candidate's CV using local text extraction and FAISS-based vector indexing.
2. **Dynamic Interviewing**: Instead of asking generic questions, the AI generates **targeted technical questions** precisely where the JD and CV differ. For example, if a JD requires AWS but the CV only mentions Azure, the AI will specifically probe for transferable cloud skills.
3. **Targeted Assessment**: The AI conducts a 3-question technical screening with the candidate in real-time, simulating a human recruiters' thought process.
4. **Expert Evaluation**: Once the interview is complete, a separate "Evaluator Agent" reviews the entire transcript against the JD requirements and provides a graded report (1-10) with actionable hiring recommendations.

## 🚀 Key Features
- **Intelligent Question Generation**: Analyzes the JD and CV to ask role-specific competency questions.
- **FAISS Vector Search**: Uses local vector storage for fast and efficient document retrieval.
- **Automated Evaluation**: Generates a detailed 1-10 score report with strengths, gaps, and recommendations.
- **Cyber-Midnight UI**: A stunning, high-contrast futuristic interface built with Streamlit.

## 🛠️ Tech Stack
- **Backend**: Python 3.9+
- **LLM**: LLaMA 3.1 8B (via Groq API)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Frontend**: Streamlit (with custom CSS/HTML injection)
<img width="1339" height="633" alt="inter2" src="https://github.com/user-attachments/assets/1349bd1f-52e4-409f-8f41-3105d771d983" />
<img width="1034" height="483" alt="inter2 1" src="https://github.com/user-attachments/assets/54d145e9-4aef-4023-8ecf-71eb6d3b1917" />
<img width="1360" height="625" alt="inter3" src="https://github.com/user-attachments/assets/6d3b4c6e-71e7-47e3-beb9-a652f37a8246" />
<img width="1357" height="567" alt="inter4" src="https://github.com/user-attachments/assets/dcd81324-4643-4ae7-ac24-a5335810a629" />


## 📦 Installation & Setup

### 1. Pre-requisites
- Python installed on your machine.
- A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-interview-agent.git
cd ai-interview-agent
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory and add your API key:
```env
GROQ_API_KEY=your_actual_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

## 🎯 How to Use
1. **Launch**: Open the app in your browser (usually `localhost:8501`).
2. **Setup**: Enter the Candidate Name and upload the Job Description & CV.
3. **Interview**: The AI will ask 3 targeted questions based on the documents.
4. **Result**: View a detailed evaluation report instantly after the interview.


