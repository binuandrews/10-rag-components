"""
02_token_splitter.py
--------------------
Concept: TokenTextSplitter

Goal:
Split text based on token count instead of characters.

LangChain Concept:
Token-aware chunking — great for OpenAI models.
"""

import os
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import TextLoader

# ==========================================
# 1️. Load Text
# ==========================================
path = "02_text_splitters/sample_texts/long_article.txt"

loader = TextLoader(path, encoding="utf-8")
docs = loader.load()

# ==========================================
# 2️. Token Splitter
# ==========================================
splitter = TokenTextSplitter(
    chunk_size=20,  # tokens
    chunk_overlap=5,
)

chunks = splitter.split_documents(docs)

# ==========================================
# 3️. Display Results
# ==========================================
print("\n TOKEN TEXT SPLITTER DEMO\n" + "-" * 40)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk.page_content}\n")

print(f"✅ Created {len(chunks)} token-based chunks.\n")

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ TokenTextSplitter is ideal when model token limits matter.
2️⃣ More reliable for multilingual text.
3️⃣ Prefer this before embedding with OpenAI or Gemini models.
""")