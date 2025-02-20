# Setup
1. Python
> Python 3.13.1
2. Tạo môi trường env
> py -m venv venv
3. Active venv (mục đích sử dụng thư viên riêng biệt).
> .\venv\Scripts\activate 
4. Install các thư viên
> cd Langchain_RAG

> pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

> pip install -r setup.txt
5. Tải model để vào folder models
> https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/blob/main/codellama-7b.Q8_0.gguf

> https://huggingface.co/RichardErkhov/vilm_-_vinallama-7b-chat-gguf/blob/main/vinallama-7b-chat.Q8_0.gguf

> https://huggingface.co/caliex/all-MiniLM-L6-v2-f16.gguf/blob/main/all-MiniLM-L6-v2-f16.gguf

6. Tải ollama (NEW)

Window: https://ollama.com/download/windows

MacOS: https://ollama.com/download/mac

7. Pull Deepseek
ollama run deepseek-r1:1.5b
ollama run deepseek-r1:7b

# Chạy 
> Folder Langchain_RAG/get_insight/
```cmd
# Đặt câu hỏi về sigma
python .\get_insight\bot11.py 
```

# Đặt prompt trên browser
url: http://127.0.0.1:7860

# URL
https://docs.omni.co/docs

https://help.sigmacomputing.com/docs/get-around-in-sigma
