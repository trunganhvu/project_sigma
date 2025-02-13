# Deepseek
import os
from typing import List, Dict
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.schema import Document

# Configuration
MODEL_FILE = "models/codellama-7b.Q8_0.gguf"
VECTOR_DB_PATHS = {
    "omni": "vector_dbs/omni_doc_db",
    "sigma": "vector_dbs/sigma_doc_db"
}
OUTPUT_DIR = "./results"
COMPARISON_CRITERIA = [
    {
        "question": "List key product features with specific examples and sources",
        "analysis_aspect": "Product Features"
    },
    {
        "question": "What trending features in data analytics industry does this product offer? Provide implementation examples",
        "analysis_aspect": "Trending Features"
    },
    {
        "question": "List pricing models and cost structures with specific numbers",
        "analysis_aspect": "Pricing Analysis"
    },
    {
        "question": "What are the unique selling points with customer case studies?",
        "analysis_aspect": "Competitive Advantages"
    }
]

class CompanyAnalyzer:
    def __init__(self):
        self.llm = CTransformers(
            model=MODEL_FILE,
            model_type="llama",
            max_new_tokens=1024,
            temperature=0.01
        )
        self.vector_dbs = self._load_vector_dbs()
        
    def _load_vector_dbs(self):
        """Load both vector databases with error handling"""
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        dbs = {}
        
        for company, path in VECTOR_DB_PATHS.items():
            try:
                dbs[company] = FAISS.load_local(
                    path,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading {company} DB: {e}")
        return dbs

    def _retrieve_info(self, company: str, query: str) -> Dict:
        """Retrieve information with source documents"""
        if company not in self.vector_dbs:
            return {"answer": "No data", "sources": []}
            
        retriever = self.vector_dbs[company].as_retriever(
            search_kwargs={"k": 5, "score_threshold": 0.7}
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        response = qa_chain.invoke({"query": query})
        return {
            "answer": response["result"],
            "sources": self._process_sources(response["source_documents"])
        }

    def _process_sources(self, docs: List[Document]) -> List[str]:
        """Extract and format sources from document metadata"""
        sources = []
        for doc in docs:
            if "source" in doc.metadata:
                # Convert filename back to original URL format
                source_url = doc.metadata["source"]
                source_url = source_url.replace("_", "/").replace(".txt", "")
                sources.append(source_url)
        return list(set(sources))[:3]  # Return top 3 unique sources

    def _generate_comparison(self, aspect: str, responses: Dict) -> str:
        """Generate structured comparison for each aspect"""
        comparison = f"\n\n=== {aspect.upper()} ===\n"
        
        for company in ["omni", "sigma"]:
            data = responses[company]
            comparison += f"\n**{company.upper()}**:\n- {data['answer']}\n"
            if data["sources"]:
                comparison += "  Sources:\n"
                for src in data["sources"]:
                    comparison += f"  - {src}\n"
        
        # Add comparative analysis
        comparison += "\n**Key Differences**:\n"
        comparison += self._analyze_differences(aspect, responses)
        return comparison

    def _analyze_differences(self, aspect: str, responses: Dict) -> str:
        """Generate AI-powered comparative analysis"""
        prompt = f"""Compare these two company responses about {aspect}. 
        Identify 3-5 key differences in this format:
        - [Difference description]
          Evidence: [Company name] ([source link])
          Impact: [Why this matters]

        Omni response: {responses['omni']['answer']}
        Sigma response: {responses['sigma']['answer']}"""
        
        analysis = self.llm.invoke(prompt)
        return analysis

    def generate_report(self, output_filename: str):
        """Generate full comparison report"""
        report_content = ""
        
        for criteria in COMPARISON_CRITERIA:
            responses = {
                company: self._retrieve_info(company, criteria["question"])
                for company in ["omni", "sigma"]
            }
            report_content += self._generate_comparison(
                criteria["analysis_aspect"],
                responses
            )
        
        # Save report
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"Report generated: {output_path}")

if __name__ == "__main__":
    analyzer = CompanyAnalyzer()
    output_filename = input("Enter report name (without extension): ") + ".txt"
    analyzer.generate_report(output_filename)