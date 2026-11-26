"""
01_faiss_basic.py
-----------------
Concept: FAISS Vector Store (Local)

Goal:
Show how to create embeddings and store them in FAISS for similarity search.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.embedding_utils import get_embedding_model

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("\n📦 FAISS VECTOR STORE DEMO\n" + "-" * 40)

# ==========================================
# 1️⃣ Create sample documents
# ==========================================
docs = [
    Document(page_content="LangChain is a framework for building LLM-powered apps."),
    Document(page_content="FAISS is a vector database developed by Meta for fast similarity search."),
    Document(page_content="Chroma is a lightweight vector database used in LangChain demos."),
]

# ==========================================
# 2️⃣ Create embeddings
# ==========================================
embedding_model = get_embedding_model("local")

# ==========================================
# 3️⃣ Create FAISS store
# ==========================================
vectorstore = FAISS.from_documents(docs, embedding_model)
print("✅ FAISS store created successfully.")

# ==========================================
# 4️⃣ Run a similarity search
# ==========================================
query = "What is LangChain used for?"
results = vectorstore.similarity_search(query, k=3)

print("\n🔍 Query:", query)
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")

# ==========================================
# 5️⃣ Save & Reload (FAISS supports persistence)
# ==========================================
vectorstore.save_local("03_vectorstores/faiss_index")
print("\n💾 Saved FAISS index to disk.")

reloaded = FAISS.load_local(
    "03_vectorstores/faiss_index", embedding_model, allow_dangerous_deserialization=True
)
print("♻️ Reloaded FAISS index successfully.")

# ==========================================
# 6️⃣ Validate reload
# ==========================================
res2 = reloaded.similarity_search("What is LangChain used for?",k=3)
print("\nRequery after reload:")
for doc in res2:
    print("→", doc.page_content)

print("""
----------------------------------------
📘 Key Takeaways:
1️⃣ FAISS provides fast local similarity search.
2️⃣ Supports saving/loading index files for reuse.
3️⃣ Ideal for CPU-based, local RAG prototypes.
""")



# If you fail to run above code:
# 1. Check if provider is set to openai in embedding_utils.py file
# 2. Try these 2 commands:
#     pip uninstall -y torch torchvision torchaudio
#     pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html