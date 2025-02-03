import os
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Configuration
model_file = "models/codellama-7b.Q8_0.gguf"
omni_vector_db_path = "vector_dbs/omni_doc_db"
sigma_vector_db_path = "vector_dbs/sigma_doc_db"
output_dir = "./results"  # Directory for storing results

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Load vector databases
def load_vector_db(vector_db_path):
    try:
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        print(f"Error loading vector database from {vector_db_path}: {e}")
        return None

# Load LLM
def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )

# Compare company information and save results
def compare_companies(question, output_filename):
    omni_db = load_vector_db(omni_vector_db_path)
    sigma_db = load_vector_db(sigma_vector_db_path)
    if not omni_db or not sigma_db:
        return

    # Create retrievers
    omni_retriever = omni_db.as_retriever(search_kwargs={"k": 3})
    sigma_retriever = sigma_db.as_retriever(search_kwargs={"k": 3})

    # Retrieve information from both databases
    llm = load_llm()
    omni_qa = RetrievalQA.from_chain_type(llm=llm, retriever=omni_retriever)
    sigma_qa = RetrievalQA.from_chain_type(llm=llm, retriever=sigma_retriever)

    # Query information
    omni_response = omni_qa.invoke({"query": question})
    sigma_response = sigma_qa.invoke({"query": question})

    # Generate comparison content
    result_text = (
        f"Question: {question}\n\n"
        f"--- Information from Omni ---\n{omni_response}\n\n"
        f"--- Information from Sigma ---\n{sigma_response}\n\n"
        f"--- Strengths and Weaknesses Comparison ---\n"
        f"Based on the retrieved data, we can summarize the differences between the two companies as follows:\n"
        f"- (This section can be further edited based on the data)\n"
    )

    # Save to a file in the ./results/ directory
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(result_text)

    print(f"Results saved to: {output_path}")

# Get user input
if __name__ == "__main__":
    question = input("Enter your question: ")
    output_filename = input("Enter result file name (without .txt extension): ") + ".txt"
    compare_companies(question, output_filename)
