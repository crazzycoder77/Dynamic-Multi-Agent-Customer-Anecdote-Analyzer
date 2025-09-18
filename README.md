# Dynamic Multi-Agent RAG with ChatGroq + Supervisor

![Python](https://img.shields.io/badge/python-3.10+-blue)

## Project Overview

This project is a **Dynamic Multi-Agent Retrieval-Augmented Generation (RAG)** system using **ChatGroq** and **LangGraph**.  
It retrieves, classifies, and aggregates customer reviews, providing insights and supervisor-guided query improvements.

**Key Features:**
- Multi-agent architecture for query expansion, retrieval, classification, aggregation, and supervision.
- Intelligent supervisor node to suggest improved queries.
- FAISS-based embedding index for fast semantic search.
- CLI output with full review text, sentiment analysis, and insights.

## Author

**Praveen Mishra**  
LinkedIn: [Your LinkedIn URL]  
GitHub: [Your GitHub URL]  

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Make sure Python 3.10+ is installed.

Install dependencies (the script also auto-installs missing packages):

bash
Copy code
pip install -r requirements.txt
Usage
Set your Groq API key:

bash
Copy code
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
Run the script:

bash
Copy code
python main.py --api_key YOUR_GROQ_API_KEY --csv path/to/reviews.csv
Optional flag to rebuild embeddings/index:

bash
Copy code
python main.py --api_key YOUR_GROQ_API_KEY --csv path/to/reviews.csv --rebuild
Enter queries in the CLI:

text
Copy code
Query> high shipping fees
Type exit or quit to terminate.

Project Structure
python
Copy code
.
├── main.py                # Main script
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── LICENSE                # License file
Example
Query: customer not happy about quality

yaml
Copy code
Total Matching Reviews: 15

ASIN Counts:
  - B0CK1WB7KC: 5
  - B0150QQ35Y: 10

Example Reviews:
ASIN: B0CK1WB7KC
Sentiment: negative
Reason: Shipping took longer than expected
Review Text: The shipping was extremely slow, arrived 10 days late...

Insights Summary:
  - Summary: Customers are frequently complaining about delayed shipping.
  - Patterns: Slow delivery times affect multiple ASINs across product categories.

Supervisor Review:
Query: high shipping fees
Issues identified:
  - No major issues identified by supervisor.
Action Suggested: OK
Future Enhancements
Support for multi-language reviews.

Integration with larger language models for deeper insights.

Web-based dashboard for visualizing insights.

Batch processing for large datasets.

License
This project is licensed under the MIT License – see the LICENSE file for details.
