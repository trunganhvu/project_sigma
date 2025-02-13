# Deepseek
import os
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from typing import List, Dict, Tuple

# Configuration
model_file = "models/codellama-7b.Q8_0.gguf"
vector_dbs = {
    "omni": "vector_dbs/omni_doc_db",
    "sigma": "vector_dbs/sigma_doc_db"
}
output_dir = "./results"
comparison_criteria = [
    ("Unique product features compared to competitor", "unique_features"),
    ("Pricing structure and cost comparison", "pricing"),
    ("Industry trending features implemented", "trending_features"),
    ("Core product differentiators", "differentiators"),
    ("Product gaps compared to competitor", "product_gaps"),
    ("Technology keywords usage frequency", "tech_keywords"),
    ("Product description depth", "description_depth"),
]

def load_vector_db(vector_db_path: str):
    try:
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        return FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading DB {vector_db_path}: {e}")
        return None

def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=2048,
        temperature=0.1,
        context_length=4096
    )

def query_company(qa_chain, query: str) -> Tuple[str, List[Dict]]:
    result = qa_chain.invoke({"query": query})
    sources = [doc.metadata.get('source', 'unknown') for doc in result['source_documents']]
    return result['result'], sources

def analyze_feature(feature: str, company_data: Dict, competitor: str):
    analysis_prompt = f"""Analyze this feature description focusing on:
    - Key differentiators with **bold** keywords
    - Evidence from source documents
    - Competitive advantages/disadvantages vs {competitor}
    - Pricing implications if mentioned
    
    Feature: {feature}"""
    return analysis_prompt

def generate_comparison_report(responses: Dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("## Omni vs Sigma Detailed Product Comparison\n\n")
        
        for criterion, key in comparison_criteria:
            f.write(f"### {criterion}\n")
            
            omni_res = responses['omni'][key]
            sigma_res = responses['sigma'][key]
            
            # Feature comparison
            f.write("**Omni Analysis**\n")
            f.write(f"{omni_res['analysis']}\nSources: {', '.join(list(set(omni_res['sources']))[:3])}\n\n")
            
            f.write("**Sigma Analysis**\n")
            f.write(f"{sigma_res['analysis']}\nSources: {', '.join(list(set(sigma_res['sources']))[:3])}\n\n")
            
            # Generate contrast
            contrast_prompt = f"""Contrast these two analyses highlighting:
            - **Key strengths** with product names
            - Feature gaps with specific examples
            - Price differences if any
            - Supported by document evidence
            
            Omni: {omni_res['analysis']}
            Sigma: {sigma_res['analysis']}"""
            
            f.write(f"**Key Contrasts**\n{self.llm.invoke(contrast_prompt)}\n\n")

def compare_companies():
    llm = load_llm()
    responses = {"omni": {}, "sigma": {}}

    for company in vector_dbs:
        db = load_vector_db(vector_dbs[company])
        if not db: continue
        
        retriever = db.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.7})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        for criterion, key in comparison_criteria:
            query = f"""When comparing to competitors, detail:
            - Specific **product features** and unique capabilities
            - Pricing models with exact numbers if available
            - Industry trends addressed
            - Technical specifications
            - Supported by document evidence
            
            Focus on: {criterion}"""
            
            response, sources = query_company(qa_chain, query)
            analysis = llm(analyze_feature(response, company, "competitor"))
            
            responses[company][key] = {
                "raw": response,
                "analysis": analysis,
                "sources": sources
            }

    # Generate final report
    output_path = os.path.join(output_dir, "bot8.md")
    generate_comparison_report(responses, output_path)
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    compare_companies()