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
Copy code
pip install -r requirements.txt
Ensure you have your Groq API key ready.
```
3. Usage
Run the program via CLI:
```
bash
Copy code
python main.py --api_key YOUR_GROQ_API_KEY [--csv path/to/reviews.csv] [--rebuild]
```
4. Arguments
--api_key (required): Your Groq API key for ChatGroq LLM access.

--csv (optional): Path to CSV file containing reviews (required if building/rebuilding embeddings).

--rebuild (optional): Force rebuild of embeddings and FAISS index from CSV.

5. Example:
```
bash
Copy code
python main.py --api_key abc123 --csv data/reviews.csv --rebuild
```
You will be prompted for queries:

text
Copy code
Query> high shipping fees
Type your query to retrieve matching reviews, insights, and trends.

Type exit or quit to stop the program.

CSV Format
The input CSV should include:

Column Name	Description
ASIN	Product identifier
Review Text	Customer review text

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
FAISS – Efficient similarity search

SentenceTransformers – Embeddings

LangGraph – Multi-agent workflow

ChatGroq – LLM-powered query expansion and classification
