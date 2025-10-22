# Akbank GenAI Bootcamp - RAG Chatbot Projesi

## 📋 Proje Amacı
LangChain dokümantasyonu üzerinde RAG (Retrieval Augmented Generation) tabanlı soru-cevap chatbotu geliştirmek.

## 🗃️ Veri Seti
- **Kaynak:** LangChain resmi GitHub repository README.md
- **İçerik:** LangChain framework'ünün tanımı, kullanım alanları, kurulum ve özellikleri
- **Boyut:** ~10KB metin verisi

## 🛠️ Kullanılan Teknolojiler
- **Vector DB:** FAISS
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Framework:** LangChain
- **Web Arayüz:** Streamlit
- **Model:** Semantic Search + Rule-based Response Generation

## 🏗️ Çözüm Mimarisi
1. **Veri İşleme:** LangChain TextLoader → RecursiveTextSplitter
2. **Embedding:** HuggingFace embeddings ile vektörleştirme
3. **Storage:** FAISS vector database
4. **Retrieval:** Benzerlik araması ile en alakalı 3 doküman
5. **Response:** Kural-tabanlı yanıt oluşturma

## 📊 Sonuçlar
- Başarılı şekilde LangChain hakkında soruları yanıtlıyor
- Hızlı response süresi (< 3 saniye)
- Temiz ve kullanıcı-dostu arayüz

## 🚀 Demo
[[Streamlit App Linki](https://akbank-genai-projesi.streamlit.app/)]

## 🔧 Kurulum
```bash
git clone [repo-link]
cd Akbank-GenAI-Projesi
pip install -r requirements.txt
streamlit run app.py
