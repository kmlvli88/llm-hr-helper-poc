# llm-hr-helper-poc

本專案為一個 HR 請假制度查詢小助手的概念驗證（POC），整合文件搜尋、員工資料與大型語言模型（LLM），達成類似 RAG（Retrieval-Augmented Generation）的架構。

##  專案架構圖
![image](https://github.com/user-attachments/assets/8f8ea6ac-a232-4cc3-bafa-9f2447cb7576)
資料來源 : aws

##  使用技術

- **PDF 解析**：`PyMuPDF (fitz)` 用於分段抽取請假規章
- **Chunk 切分**：每 300 字切割，儲存為 JSON
- **向量嵌入模型**：`sentence-transformers/all-MiniLM-L6-v2`
- **向量索引搜尋**：使用 `FAISS` 建立 L2 索引
- **本地大語言模型推論**：透過 `Ollama` 呼叫 `mistral` 模型生成回答
- **Prompt 注入**：將請假規章 + 員工請假紀錄餵給 LLM，讓 AI 回答人性化內容


學習順序:
1.列舉模型種類，認識每個模型特性，再配合需求去應用
2.決定服務該用哪種模型，


