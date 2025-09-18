# Dynamic Multi-Agent Customer Review Analyzer

🚀 **Dynamic Multi-Agent AI system** to analyze large volumes of customer reviews.  
It uses **FAISS embeddings**, **LangGraph**, and **ChatGroq** to intelligently retrieve, classify, and aggregate reviews to provide actionable insights.  

The system includes a **supervisor agent** that iteratively improves queries for better relevance.

---

## Features

- 🔍 **Query Expansion** using LLM (ChatGroq)
- 🔎 **Dynamic Retrieval** from a FAISS index of reviews
- 📝 **Classification** of reviews by match, sentiment, and reason
- 🧩 **Aggregation** of insights and trends across products
- 🛠️ **Supervisor Agent** for improved query suggestions
- 🖥️ **CLI Interface** for interactive querying

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/crazzycoder77/Dynamic-Multi-Agent-Customer-Anecdote-Analyzer.git
cd Dynamic-Multi-Agent-Customer-Anecdote-Analyzer
```
2. Install required dependencies:
```
bash
pip install -r requirements.txt
```
Ensure you have your Groq API key ready.

##  Usage
Run the program via CLI:
```
bash
python main.py --api_key YOUR_GROQ_API_KEY [--csv path/to/reviews.csv] [--rebuild]
```
## Arguments
--api_key (required): Your Groq API key for ChatGroq LLM access.

--csv (optional): Path to CSV file containing reviews (required if building/rebuilding embeddings).

--rebuild (optional): Force rebuild of embeddings and FAISS index from CSV.

##  Example:
```
bash
Copy code
python main.py --api_key abc123 --csv data/reviews.csv --rebuild
```
You will be prompted for queries:
```
Query> high shipping fees
```
Type your query to retrieve matching reviews, insights, and trends.

Type exit or quit to stop the program.

## CSV Format
The input CSV should include:

| Column  | Name	Description |
| ------------- | ------------- |
| ASIN  | Product identifie  |
| Review  | Text	Customer review text  |

## License
This project is licensed under the MIT License. See the <a href="https://github.com/crazzycoder77/Dynamic-Multi-Agent-Customer-Anecdote-Analyzer/blob/main/LICENSE.txt">LICENSE</a> file for details.

## Acknowledgements
<a href="https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/">FAISS</a> – Efficient similarity search

<a href="https://sbert.net/">SentenceTransformers</a> – Embeddings

<a href="https://www.langchain.com/langgraph">LangGraph</a> – Multi-agent workflow

<a href="https://groq.com/">ChatGroq</a> – LLM-powered query expansion and classification
