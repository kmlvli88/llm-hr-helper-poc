import os
import fitz  # type: ignore
import ollama  # type: ignore
import faiss  # type: ignore
import json
from sentence_transformers import SentenceTransformer  # type: ignore
import numpy as np

# --- STEP 1: 抽 PDF Chunk ---
def extract_chunks_from_pdf(pdf_path, chunk_size=300):
    doc = fitz.open(pdf_path)
    full_text = ''.join([page.get_text("text") for page in doc])
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# --- STEP 2: 儲存 / 載入 Chunk ---
def load_or_create_chunks(pdf_path, chunk_path, chunk_size=300):
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            print("🔹 已載入儲存的 chunk")
            return json.load(f)
    else:
        print("📄 正在從 PDF 建立 chunk...")
        chunks = extract_chunks_from_pdf(pdf_path, chunk_size)
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        return chunks

# --- STEP 3: 建立或載入向量索引 ---
def load_or_create_faiss(chunks, emb_path, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)

    if os.path.exists(emb_path):
        print("🔹 已載入儲存的向量與索引")
        embeddings = np.load(emb_path)
    else:
        print("🔎 正在建立向量嵌入...")
        embeddings = model.encode(chunks, convert_to_numpy=True)
        np.save(emb_path, embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, model, embeddings

# --- STEP 4: 查找最相近段落 ---
def search_similar_chunks(question, index, model, chunks, top_k=1):
    question_vector = model.encode([question], convert_to_numpy=True)
    D, I = index.search(question_vector, top_k)
    return [chunks[i] for i in I[0]]

# --- 使用者假資料（可替換為 JSON 讀取） ---
employee_info = {
    "name": "員工A",
    "employee_id": "00048377",
    "personal_leave_taken": 5,
    "personal_leave_limit": 14,
    "annual_leave_taken": 3,
    "annual_leave_total": 10,
    "marriage_leave_remaining": 2
}

# --- 設定路徑 ---
pdf_path = "D:\\HR小助手_POC\\06-06『請假、假別規定』20170101附件五(TW).pdf"
chunk_path = "chunks.json"
emb_path = "embeddings.npy"

# --- 載入 / 建立 chunks + embedding + index ---
chunks = load_or_create_chunks(pdf_path, chunk_path)
index, embed_model, embeddings = load_or_create_faiss(chunks, emb_path)

# --- 使用者輸入問題 ---
user_question = input("請輸入你的問題：")
relevant_chunks = search_similar_chunks(user_question, index, embed_model, chunks)

# --- 建立 Prompt（只用 top 1 段） ---
system_prompt = f"""
你是公司的 HR AI 助理，請使用繁體中文根據以下「請假政策」與「員工資料」回答問題，並且以簡潔清楚的方式回覆。

【請假政策】:
{relevant_chunks[0]}

【員工請假紀錄】:
姓名：{employee_info['name']}
事假：{employee_info['personal_leave_taken']} / {employee_info['personal_leave_limit']} 天
特休：{employee_info['annual_leave_taken']} / {employee_info['annual_leave_total']} 天
婚假剩餘：{employee_info['marriage_leave_remaining']} 天
"""

# --- 呼叫 LLM ---
response = ollama.chat(model='mistral', messages=[
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_question}
])

# --- 輸出回答 ---
print("\n🧠 HR AI 回覆：")
print(response['message']['content'])
