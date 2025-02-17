import os
import gradio as gr
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Configuration
model_file = "models/codellama-7b.Q8_0.gguf"
vector_db_path = "vector_dbs/omni_doc_db"  # Sử dụng một vector database duy nhất

# Load the vector database
def load_vector_db():
    try:
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        return FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector database: {str(e)}")
        return None

# Load the language model
def load_llm():
    return CTransformers(
        model=model_file,
        model_type="llama",
        config={'max_new_tokens': 1024, 'temperature': 0.01, 'context_length': 4096}
    )

def format_question(question: str) -> str:
    template = (
        "You are an AI assistant. Answer the question strictly based on the provided context.\n"
        "If the answer cannot be found in the context, respond with 'Information not found in the provided sources.'\n"
        "Be concise and provide references to the source documents.\n\n"
        "Question: {question}\n"
        "Context:"
    )
    return template.format(question=question)

# Function to ask a question and get an answer
def ask_question(question: str) -> str:
    db = load_vector_db()
    if not db:
        return "Error: Unable to load the vector database."

    llm = load_llm()
    retriever = db.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.7})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    formatted_question = format_question(question)
    response = qa.invoke({"query": formatted_question })
    answer = response['result']
    sources = [doc.metadata.get('source', 'Unknown') for doc in response['source_documents']]
    formatted_sources = "\n".join(f"- {os.path.basename(src)}" for src in set(sources))
    return f"**Answer:** {answer}\n\n**Sources:**\n{formatted_sources if formatted_sources else 'No sources found.'}"

# Gradio interface
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="RAG Chatbot: Single Vector Database",
    description="Ask any question about the loaded documents. Powered by a local language model and FAISS."
)

if __name__ == "__main__":
    interface.launch()
