# Deepseek
import os
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from typing import List, Tuple
from langchain.prompts import PromptTemplate

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
    
    retriever = db.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.7})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa.invoke({"query": question})
    return response['result'], [doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]

def format_sources(sources: List[str]) -> str:
    unique_sources = list(set(sources))
    return "\n".join(f"• {os.path.basename(src).replace('_', '/')}" for src in unique_sources) if unique_sources else "No sources found"

def comparative_analysis(omni_data: Tuple[str, List[str]], sigma_data: Tuple[str, List[str]]) -> str:
    def parse_response(response: str) -> Tuple[List[str], List[str]]:
        strengths = []
        weaknesses = []
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            if 'strength' in line.lower():
                current_section = 'strength'
            elif 'weakness' in line.lower():
                current_section = 'weakness'
            
            if current_section == 'strength' and line and ':' in line:
                strengths.extend(line.split(':')[1].split(';'))
            elif current_section == 'weakness' and line and ':' in line:
                weaknesses.extend(line.split(':')[1].split(';'))
        
        return [s.strip() for s in strengths if s.strip()], [w.strip() for w in weaknesses if w.strip()]

    omni_str, omni_weak = parse_response(omni_data[0])
    sigma_str, sigma_weak = parse_response(sigma_data[0])

    comparison = ["\n=== Detailed Comparison ==="]
    
    # Strength analysis
    comparison.append("\n🔍 Core Strengths Comparison:")
    comparison.append("\nOmni.co excels in:")
    for idx, s in enumerate(omni_str[:3]):
        comparison.append(f"{idx+1}. {s} (Verified in: {format_sources(omni_data[1][:2])})")
    
    comparison.append("\nSigmacomputing.com stands out with:")
    for idx, s in enumerate(sigma_str[:3]):
        comparison.append(f"{idx+1}. {s} (Verified in: {format_sources(sigma_data[1][:2])})")

    # Weakness analysis
    comparison.append("\n\n⚠️ Notable Weaknesses:")
    comparison.append("\nOmni.co limitations:")
    for idx, w in enumerate(omni_weak[:2]):
        comparison.append(f"- {w} (Source: {format_sources(omni_data[1][-1:])})")
    
    comparison.append("\nSigmacomputing.com limitations:")
    for idx, w in enumerate(sigma_weak[:2]):
        comparison.append(f"- {w} (Source: {format_sources(sigma_data[1][-1:])})")

    return '\n'.join(comparison)

comparison_prompt = PromptTemplate(
    input_variables=["omni_info", "sigma_info"],
    template="""
    You are a highly skilled business analyst performing a competitive analysis between Omni and Sigma.  Your goal is to provide a structured, in-depth comparison focusing on product features, strengths, weaknesses, pricing (if available), and actionable insights.

    Specifically:

    1.  **Product Features:** Identify and compare key product features, highlighting trending, unique, and valuable aspects. Provide concrete examples and evidence from the provided context to support your claims.  For example, "Omni offers feature X, which is similar to Sigma's feature Y, but Omni's implementation includes Z, making it more efficient for users."

    2.  **Strengths and Weaknesses:**  For each company, identify 3 significant strengths and 2 key weaknesses.  Be specific!  Instead of "good customer support," say "Omni's customer support has a median response time of under 2 hours, based on their website testimonials." Provide supporting evidence.

    3.  **Pricing:** If pricing information is available in the context, compare the pricing models, highlighting any differences and explaining their potential impact on different customer segments.  If pricing is unavailable, mention that.

    4.  **Actionable Insights:** Conclude with 3 actionable insights for each company, based on your analysis. These should be concrete recommendations on how they can leverage their strengths, address their weaknesses, or capitalize on market opportunities.

    Omni Information:
    {omni_info}

    Sigma Information:
    {sigma_info}

    Comparison and Analysis:
    """
)

def comparative_analysis(omni_data: Tuple[str, List[str]], sigma_data: Tuple[str, List[str]], llm) -> str:
    omni_info = omni_data[0] + "\n\nSources:\n" + format_sources(omni_data[1])
    sigma_info = sigma_data[0] + "\n\nSources:\n" + format_sources(sigma_data[1])

    comparison_chain = RetrievalQA.from_chain_type(llm=llm, retriever=None, chain_type_kwargs={"prompt": comparison_prompt})
    comparison_result = comparison_chain.invoke({"omni_info": omni_info, "sigma_info": sigma_info})

    return comparison_result['result']

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

    # Generate comparison report
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
            (results['sigma']['response'], results['sigma']['sources']),
            llm  # Pass the LLM to the comparison function
        )
    ]

    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    question = """Find, analyze and compare the key strengths and weaknesses of our company. 
    Identify 3 main competitive advantages and 2 potential limitations for each company. 
    Include specific product features and market positioning details."""
    output_filename = "omni_vs_sigma_comparison2.md"
    generate_report(question, output_filename)