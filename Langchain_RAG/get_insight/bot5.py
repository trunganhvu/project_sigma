# ai studio
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
    omni_response, omni_sources = omni_data
    sigma_response, sigma_sources = sigma_data

    comparison = ["\n\n# 📊 Detailed Comparative Analysis"]

    # --- Product Features ---
    comparison.append("\n## 🚀 Product Features & Capabilities")

    # Omni Features Analysis
    comparison.append("\n### Omni.co: Product Strengths and Unique Features")
    omni_product_features = analyze_product_features(omni_response)
    if omni_product_features:
        for feature, details in omni_product_features.items():
            comparison.append(f"\n**Feature:** {feature}")
            if details['description']:
                comparison.append(f"- **Description:** {details['description']}")
            if details['example']:
                comparison.append(f"- **Example/Evidence:** {details['example']} (Source: {format_sources(details['sources'][:2])})") # Limit sources to 2 for brevity
    else:
        comparison.append("\nNo specific product features identified in analysis.")


    # Sigma Features Analysis
    comparison.append("\n### Sigmacomputing.com: Product Strengths and Unique Features")
    sigma_product_features = analyze_product_features(sigma_response)
    if sigma_product_features:
        for feature, details in sigma_product_features.items():
            comparison.append(f"\n**Feature:** {feature}")
            if details['description']:
                comparison.append(f"- **Description:** {details['description']}")
            if details['example']:
                comparison.append(f"- **Example/Evidence:** {details['example']} (Source: {format_sources(details['sources'][:2])})") # Limit sources to 2 for brevity
    else:
        comparison.append("\nNo specific product features identified in analysis.")

    # --- Trending Features ---
    comparison.append("\n## ✨ Trending Features in the Industry")
    trending_features_comparison = compare_trending_features(omni_response, sigma_response)
    if trending_features_comparison:
        comparison.append(trending_features_comparison)
    else:
        comparison.append("\nNo comparative analysis on trending features available.")


    # --- Pricing Comparison ---
    comparison.append("\n## 💰 Pricing and Value Proposition")
    pricing_comparison = compare_pricing(omni_response, sigma_response)
    if pricing_comparison:
        comparison.append(pricing_comparison)
    else:
        comparison.append("\nNo pricing comparison data available in the analysis.")


    # --- Overall Strengths and Weaknesses Summary ---
    comparison.append("\n## 🏆 Overall Strengths and Areas for Improvement")

    # Omni Strengths and Weaknesses
    comparison.append("\n### Omni.co: Strengths & Weaknesses Summary")
    omni_strengths_weaknesses = analyze_strengths_weaknesses(omni_response, omni_sources)
    comparison.extend(omni_strengths_weaknesses)

    # Sigma Strengths and Weaknesses
    comparison.append("\n### Sigmacomputing.com: Strengths & Weaknesses Summary")
    sigma_strengths_weaknesses = analyze_strengths_weaknesses(sigma_response, sigma_sources)
    comparison.extend(sigma_strengths_weaknesses)


    return '\n'.join(comparison)


def analyze_product_features(response: str) -> dict:
    """
    Parses the response to identify product features, descriptions, and examples.
    This is a placeholder and needs to be improved based on LLM output format.
    Expects the LLM to output features in a structured way.
    Example expected format from LLM (can be adjusted based on actual output):

    **Product Features:**
    - **Feature 1: Data Visualization**
        - Description: Offers interactive dashboards and charts.
        - Example: Users can create custom reports with drag-and-drop interface (Source: ...).
    - **Feature 2: Real-time Collaboration**
        - Description: Enables multiple users to work on the same data simultaneously.
        - Example: Supports concurrent editing of dashboards and datasets (Source: ...).
    """
    features = {}
    lines = response.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("**Feature:") or line.startswith("Feature:"): # Handle variations in LLM output
            feature_name = line.split(":", 1)[1].strip()
            description = None
            example = None
            sources = []

            # Look for description and example in the following lines
            i += 1
            while i < len(lines) and lines[i].strip().startswith("- **Description:**"):
                description = lines[i].strip().split(":", 1)[1].strip()
                i += 1
                break # Expect only one description

            while i < len(lines) and lines[i].strip().startswith("- **Example:") or (lines[i].strip().startswith("- Example:")) or (lines[i].strip().startswith("- **Evidence:")) or (lines[i].strip().startswith("- Evidence:")) : # Handle variations
                example_line = lines[i].strip()
                example_prefix = "- **Example:**" if example_line.startswith("- **Example:") else "- Example:" if example_line.startswith("- Example:") else "- **Evidence:**" if example_line.startswith("- **Evidence:") else "- Evidence:" # Handle variations
                example = example_line.split(example_prefix, 1)[1].strip()


                if "(Source:" in example: # Extract source from example line if present
                    example_parts = example.split("(Source:")
                    example = example_parts[0].strip()
                    source_str = example_parts[1].replace(")", "").strip()
                    sources_list = [s.strip().replace('_', '/') for s in source_str.split('•') if s.strip()] # Adjust source parsing to match format_sources
                    sources.extend(sources_list)

                i += 1
                break # Expect only one example

            features[feature_name] = {'description': description, 'example': example, 'sources': sources}
        else:
            i += 1 # Move to next line if not a feature

    return features


def compare_trending_features(omni_response: str, sigma_response: str) -> str:
    """
    Analyzes and compares trending features mentioned in the responses.
    Placeholder - needs implementation based on expected LLM output.
    Expects LLM to explicitly mention "trending features" or similar.
    """
    omni_trending = extract_trending_features(omni_response)
    sigma_trending = extract_trending_features(sigma_response)

    comparison_str = ""
    if omni_trending or sigma_trending:
        comparison_str += "\n### Comparison of Trending Features:"
        if omni_trending:
            comparison_str += "\n**Omni.co:** Mentions trending features like: " + ", ".join(omni_trending)
        else:
            comparison_str += "\n**Omni.co:** No trending features explicitly mentioned."

        if sigma_trending:
            comparison_str += "\n\n**Sigmacomputing.com:** Highlights trending features such as: " + ", ".join(sigma_trending)
        else:
            comparison_str += "\n\n**Sigmacomputing.com:** No trending features explicitly mentioned."
    return comparison_str


def extract_trending_features(response: str) -> List[str]:
    """
    Extracts trending features from the response string.
    Placeholder - adjust keyword and extraction logic as needed.
    """
    trending_features = []
    if "trending feature" in response.lower() or "industry trend" in response.lower():
        # Simple keyword based extraction - improve with more sophisticated NLP if needed
        for line in response.split('\n'):
            if "trending feature" in line.lower() or "industry trend" in line.lower():
                # Basic extraction - may need refinement
                feature = line.split("trending feature", 1)[-1].strip().strip(':').strip('.').split("industry trend", 1)[-1].strip().strip(':').strip('.')
                if feature:
                    trending_features.append(feature) # Further cleaning might be needed
    return trending_features



def compare_pricing(omni_response: str, sigma_response: str) -> str:
    """
    Compares pricing information if available in the responses.
    Placeholder - needs implementation based on expected LLM output format.
    Expects LLM to output pricing details in a comparable manner.
    """
    omni_pricing = extract_pricing_info(omni_response)
    sigma_pricing = extract_pricing_info(sigma_response)

    comparison_str = ""
    if omni_pricing or sigma_pricing:
        comparison_str += "\n### Pricing Comparison:"
        if omni_pricing:
            comparison_str += f"\n**Omni.co:** Pricing model appears to be {omni_pricing['model']}. {omni_pricing.get('details', 'Further details not available.')}"
        else:
            comparison_str += "\n**Omni.co:** No explicit pricing information found in analysis."

        if sigma_pricing:
            comparison_str += f"\n\n**Sigmacomputing.com:** Pricing seems to follow a {sigma_pricing['model']} structure. {sigma_pricing.get('details', 'Further details not available.')}"
        else:
            comparison_str += "\n\n**Sigmacomputing.com:** No explicit pricing information found in analysis."
        if omni_pricing and sigma_pricing:
            if omni_pricing['model'] != "No Information" and sigma_pricing['model'] != "No Information":
                comparison_str += analyze_price_difference(omni_pricing, sigma_pricing) # Add price difference analysis
    return comparison_str

def analyze_price_difference(omni_pricing, sigma_pricing) -> str:
    """
    Analyzes and describes the price difference if pricing models are comparable.
    Placeholder - needs more sophisticated logic if actual price points are extracted.
    Currently compares models conceptually.
    """
    diff_analysis = ""
    omni_model = omni_pricing['model'].lower()
    sigma_model = sigma_pricing['model'].lower()

    if omni_model != "no information" and sigma_model != "no information":
        if omni_model == sigma_model:
            diff_analysis += f"\n\nBoth companies seem to utilize a similar {omni_model} pricing model."
        else:
            diff_analysis += f"\n\n**Pricing Model Differences:** Omni.co uses a {omni_model} model while Sigmacomputing.com employs a {sigma_model} approach. This suggests different strategies in how they offer value and structure costs."
    return diff_analysis


def extract_pricing_info(response: str) -> dict:
    """
    Extracts pricing information from the response string.
    Placeholder - adjust keywords and extraction logic based on LLM output.
    """
    pricing_info = {"model": "No Information", "details": None}
    if "pricing" in response.lower() or "cost" in response.lower() or "subscription" in response.lower():
        if "subscription-based" in response.lower():
            pricing_info["model"] = "Subscription-based"
            # Example of extracting details - refine as needed
            if "tiers" in response.lower():
                pricing_info["details"] = "Offers tiered subscription plans."
        elif "usage-based" in response.lower():
            pricing_info["model"] = "Usage-based"
            pricing_info["details"] = "Pricing is based on usage."
        # Add more pricing model detections as needed based on data

    return pricing_info


def analyze_strengths_weaknesses(response: str, sources: List[str]) -> List[str]:
    """
    Parses the response to identify strengths and weaknesses, and formats them with sources.
    Improved parsing to be more robust and extract more detail.
    """
    strengths_weaknesses_output = []
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
            strengths.append(line.split(':', 1)[1].strip()) # Capture strength description
        elif current_section == 'weakness' and line and ':' in line:
            weaknesses.append(line.split(':', 1)[1].strip()) # Capture weakness description

    if strengths:
        strengths_weaknesses_output.append("\n#### Strengths:")
        for idx, s in enumerate(strengths[:3]): # Limit to top 3 strengths
            strengths_weaknesses_output.append(f"- {s} (Sources: {format_sources(sources[:2]) if sources else 'No sources available'})") # Add sources, limit to 2 for brevity

    if weaknesses:
        strengths_weaknesses_output.append("\n#### Areas for Improvement/Weaknesses:")
        for idx, w in enumerate(weaknesses[:2]): # Limit to top 2 weaknesses
            strengths_weaknesses_output.append(f"- {w} (Sources: {format_sources(sources[-1:]) if sources else 'No sources available'})") # Add sources, limit to 1 for weakness sources

    return strengths_weaknesses_output


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
        "# 🚀 Competitive Analysis Report: Omni.co vs Sigmacomputing.com",
        f"## 🎯 Analysis Focus: {question}",

        "\n## 🏢 Omni.co Deep Dive",
        f"**Key Analysis & Findings:**\n{results['omni']['response']}",
        f"\n**Supporting Documents:**\n{format_sources(results['omni']['sources'])}",

        "\n## 🏢 Sigmacomputing.com Deep Dive",
        f"**Key Analysis & Findings:**\n{results['sigma']['response']}",
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
    question = """
    Conduct an in-depth comparative analysis of Omni.co and Sigmacomputing.com, focusing specifically on their product offerings.

    Identify and compare:
    1. **Key Product Features:** For each company, list and describe 3-5 significant product features. Highlight any unique or innovative features. Where possible, provide examples and cite evidence from the provided web documents.

    2. **Trending Features:** Analyze if and how each company incorporates or addresses current industry trends in their products (e.g., AI-driven analytics, real-time collaboration, data governance features). Compare their approaches.

    3. **Pricing and Value:** Compare the pricing models or any available information on cost and value proposition for each company's products.  If pricing details are present, analyze any significant differences and potential reasons.

    Finally, summarize the overall strengths and weaknesses of each company's product based on your analysis, identifying 3 key strengths and 2 potential areas for improvement for each.
    """
    output_filename = "omni_vs_sigma_product_comparison.md"
    generate_report(question, output_filename)