# Finance Q&A Assistant using LoRA-Tuned Llama and Streamlit (CPU & CUDA Versions)
I built this project to explore fine-tuning a large language model (Llama 3.2: 3B) for finance-specific Q&A. I maintain two versions of the code:
1. CPU Version – for systems without CUDA or where GPU isn’t available.
2. CUDA Version – optimized for NVIDIA GPUs with CUDA, allowing faster model inference and training.

This README explains how I structured the project and how to use each version.

# Why I Have Two Versions
1. Not everyone has an NVIDIA GPU, so I made a CPU-friendly version to show the same functionality (albeit more slowly).
2. For those with CUDA, I created a version that can take advantage of GPU acceleration, especially for bigger LLMs.
3. This let me practice bitsandbytes or PEFT GPU-based approaches while still offering a CPU fallback.

# Project Overview
1. Data: I scraped and cleaned finance-related content from Wikipedia and Investopedia, then fine-tuned a Llama 3.2 model (using Hugging Face PEFT).
2. Deployment: A Streamlit application provides a web-based Q&A interface.
3. CPU vs. GPU: The main difference is how the model is loaded (device_map="auto" for GPU or an explicit CPU map if no GPU).

# Installation & Setup

1. Clone the Repo
    - git clone https://github.com/saivirinchi125/finance-qa-llm.git

2. Install Dependencies
    - pip install -r requirements.txt
   - For CPU usage only, you might ignore bitsandbytes.
   - For GPU usage, ensure CUDA is installed and bitsandbytes is included.

# Running the CPU Version
1. Navigate to the app_cpu.py.
2. Run the Streamlit app:
    - streamlit run app_cpu.py
3. Open the provided local URL (usually http://localhost:8501) in your browser.
4. Enter a prompt in the text box. The LLM will respond, but inference might be slower since it’s all running on CPU in float32 (or float16 if your CPU supports it).   

# Running the CUDA Version
1. Ensure your system has NVIDIA drivers and CUDA installed.
2. Navigate to app_cuda.py
3. Run the Streamlit app:
    - streamlit run app_cuda.py
4. Wait until it says it’s listening on port 8501 (or increments if that port is taken).
5. Open http://localhost:8501 in your browser. Now the LLM runs faster, leveraging GPU.
6. If you have bitsandbytes set up, you may see 8-bit or 4-bit logs in the console.

# Project Details
1. Scraping & Cleaning:
I used requests + BeautifulSoup for Investopedia, wikipedia library for Wikipedia.
2. Notes on Fine-Tuning:
    - PEFT (LoRA) approach to reduce VRAM usage.
    - CPU or GPU, so I can demonstrate different deployment scenarios.
    - If I only have a CPU, I set device_map={"": "cpu"} in the script, though it can be extremely slow.
    - If I’m on a GPU environment, I use device_map="auto" and possibly load_in_8bit=True with bitsandbytes for memory savings.
3. Streamlit UI:
    - A simple text area for prompts and a “Generate” button.
    - On CPU, it can take 10–30 seconds or more for bigger models.
    - On GPU (with CUDA), responses are faster.

# Differences Between CPU & CUDA Versions
1. device_map: CPU version sets device_map={"": "cpu"} or None; GPU version sets it to "auto" or a GPU ID.
2. Quantization: CPU version typically runs in float32 (or bfloat16 if your CPU has AVX512). The GPU version can use 8-bit or 4-bit quantization with bitsandbytes.
3. Speed: GPU version can be significantly faster for inference or training on large models. CPU version is slower but ensures broad compatibility.

# Usage Example
Example Flow:
1. Prompt: “Could you briefly explain what the Price-Earnings ratio is?”
2. Model (in your Streamlit app) responds with a short summary referencing your fine-tuned finance knowledge.
3. You can refine the query (e.g., “What are its limitations?”) in a conversation-like fashion (though my current script is set up for single-turn Q&A).

# Known Issues & Future Plans
1. On large models (3B+ parameters), CPU inference is quite slow. I may integrate a specialized CPU quantization approach (e.g., llama.cpp).
2. Currently, I only tested CPU Version. The GPU version requires an NVIDIA GPU. AMD GPUs would need ROCm or other setups.
3. I might integrate a vector database (FAISS or Chroma) to fetch relevant text from PDF reports.
4. Currently, the chatbot is mostly one-turn Q&A. I may add multi-turn conversation history.
5. I might collect more finance Q&A pairs to measure accuracy or do a human evaluation.
6. Possibly host on Streamlit Community Cloud or Hugging Face Spaces to share with others without them needing to install anything.

# Acknowledgements
 - Hugging Face Transformers and PEFT for the easy fine-tuning APIs.
 - Streamlit for a delightful web UI library.
 - Investopedia and Wikipedia for finance data (used under their respective licenses; for educational purposes only).
