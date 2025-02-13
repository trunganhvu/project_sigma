# GPT
import os
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from typing import List, Tuple

# Configuration
model_file = "models/codellama-7b.Q8_0.gguf"
vector_dbs = {
    "omni": "vector_dbs/omni_doc_db",
    "sigma": "vector_dbs/sigma_doc_db"
}
output_dir = "./results"
os.makedirs(output_dir, exist_ok=True)

def load_vector_db(vector_db_path: str):
    try:
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        return FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading {vector_db_path}: {str(e)}")
        return None

def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        config={'max_new_tokens': 1024, 'temperature': 0.01, 'context_length': 4096}
    )

def analyze_company(llm, db, company_name: str, question: str) -> Tuple[str, List[str]]:
    if not db:
        return "No data available", []
    
    retriever = db.as_retriever(search_kwargs={"k": 10, "score_threshold": 0.7})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa.invoke({"query": question})
    return response['result'], [doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]

def extract_features(response: str) -> Tuple[List[str], List[str]]:
    features = []
    price_details = []
    for line in response.split('\n'):
        if "feature" in line.lower() or "functionality" in line.lower():
            features.append(line.strip())
        if "price" in line.lower() or "cost" in line.lower():
            price_details.append(line.strip())
    return features, price_details

def comparative_analysis(omni_data: Tuple[str, List[str]], sigma_data: Tuple[str, List[str]]) -> str:
    omni_features, omni_prices = extract_features(omni_data[0])
    sigma_features, sigma_prices = extract_features(sigma_data[0])
    
    comparison = ["\n=== Detailed Product Comparison ==="]
    
    # Feature analysis
    comparison.append("\n🔍 Product Features Comparison:")
    comparison.append("\n**Omni.co Product Features:**")
    for idx, f in enumerate(omni_features[:5]):
        comparison.append(f"- {f}")
    
    comparison.append("\n**Sigmacomputing.com Product Features:**")
    for idx, f in enumerate(sigma_features[:5]):
        comparison.append(f"- {f}")
    
    # Price comparison
    comparison.append("\n\n💰 Price Comparison:")
    comparison.append("\n**Omni.co Pricing Details:**")
    for p in omni_prices[:3]:
        comparison.append(f"- {p}")
    
    comparison.append("\n**Sigmacomputing.com Pricing Details:**")
    for p in sigma_prices[:3]:
        comparison.append(f"- {p}")
    
    return '\n'.join(comparison)

def format_sources(sources: List[str]) -> str:
    unique_sources = list(set(sources))
    return "\n".join(f"• {os.path.basename(src).replace('_', '/')}" for src in unique_sources) if unique_sources else "No sources found"

def generate_report(question: str, filename: str):
    llm = load_llm()
    results = {}
    
    for company, path in vector_dbs.items():
        db = load_vector_db(path)
        response, sources = analyze_company(llm, db, company, question)
        results[company] = {
            'response': response,
            'sources': sources
        }

    report = [
        "# Competitive Analysis Report",
        f"## Analysis Question: {question}",
        "\n## Omni.co Analysis",
        f"**Key Findings:**\n{results['omni']['response']}",
        f"\n**Supporting Documents:**\n{format_sources(results['omni']['sources'])}",
        "\n## Sigmacomputing.com Analysis",
        f"**Key Findings:**\n{results['sigma']['response']}",
        f"\n**Supporting Documents:**\n{format_sources(results['sigma']['sources'])}",
        comparative_analysis(
            (results['omni']['response'], results['omni']['sources']),
            (results['sigma']['response'], results['sigma']['sources'])
        )
    ]

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    question = """Find and compare the key strengths and weaknesses of our company.
    Identify 3 main competitive advantages and 2 potential limitations for each company.
    Include specific product features, trending functionalities, and price details."""
    
    output_filename = "omni_vs_sigma_detailed_comparison.md"
    generate_report(question, output_filename)
