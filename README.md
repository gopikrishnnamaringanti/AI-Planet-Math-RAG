# AI-Planet-Math-RAG

AI-Planet-Math-RAG is a **math-focused AI assistant** that uses **Retrieval-Augmented Generation (RAG)** to answer mathematical queries. It leverages **Google’s Gemini LLM**, **Qdrant vector store**, and a **React frontend** to provide step-by-step explanations, numerical solutions, and context-aware answers for a wide range of math problems.

---

## Features

- Supports **math queries** including:
  - Algebra, Calculus, Linear Algebra, Trigonometry, Matrices, Limits, and more.
  - Symbolic computations and numerical solvers for complex equations.
- **Step-by-step explanations** for better understanding.
- **Context-aware retrieval** from a vector store of math documents.
- **Human feedback loop** for improving answer quality.
- **FastAPI backend** with REST endpoints.
- **React frontend** for interactive user experience.

---

## Tech Stack

- **Backend:** FastAPI, Python 3.11
- **Frontend:** React
- **AI/ML:** Google Gemini LLM via `google.generativeai`, DSPy for chain-of-thought reasoning
- **Vector Store:** Qdrant
- **Search:** Tavily search tool
- **Others:** SciPy (numerical solvers), LangChain, LangGraph, Pydantic, Dotenv

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gopikrishnnamaringanti/AI-Planet-Math-RAG.git
   cd AI-Planet-Math-RAG
