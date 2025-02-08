# Setup
1. Python
> Python 3.13.1
2. Tạo môi trường env
> py -m venv .venv
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

# Chạy 
> Folder Langchain_RAG/get_insight/
```cmd
# Đặt câu hỏi về sigma
python .\get_insight\sigma_bot.py 

# Đặt câu hỏi về omni
python .\get_insight\omni_bot.py

# Đặt câu hỏi compare về sigma và omni
python .\get_insight\bot.py
```

# URL
https://docs.omni.co/docs

https://help.sigmacomputing.com/docs/get-around-in-sigma

# TODO
1. Crawl omni
> https://omni.co/blog
> https://docs.omni.co/guides
> https://omni.co/compare
> https://omni.co/changelog
> https://community.omni.co/

2. Crawl sigma