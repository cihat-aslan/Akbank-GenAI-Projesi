import os
import re
import streamlit as st
from dotenv import load_dotenv

# ---- BASİT İMPORTLAR ----
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- .env dosyasındaki API anahtarını yükle ---
load_dotenv()

# --- AKILLI HTML TEMİZLEME ---
def clean_html_tags(text):
    """Sadece gerçek HTML tag'lerini temizle"""
    # Eğer çok fazla HTML tag'i yoksa, orijinal metni koru
    html_tag_count = len(re.findall(r'<[^>]+>', text))
    if html_tag_count < 3:  # Çok az HTML tag varsa temizleme
        return text
    
    # Çok fazla HTML tag varsa temizle
    clean_text = re.sub(r'<[^>]+>', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text.strip()

# --- 1. Veri Yükleme ve Parçalama ---
def get_text_chunks(text_file):
    try:
        loader = TextLoader(text_file, encoding="utf-8")
        documents = loader.load()
        
        # HTML tag'lerini temizle
        for doc in documents:
            doc.page_content = clean_html_tags(doc.page_content)
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        return chunks
    except Exception as e:
        st.error(f"Veri yükleme hatası: {str(e)}")
        return None

# --- 2. Vektör Deposu (FAISS) oluşturma ---
def get_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"Vektör deposu oluşturma hatası: {str(e)}")
        return None

# --- 3. TEMİZ YANIT SİSTEMİ ---
def generate_response(context, question):
    """
    LLM olmadan temiz bir yanıt oluşturucu
    """
    # Context'i temizle
    clean_context = clean_html_tags(context)
    
    # Başlık ve badge'leri atla, anlamlı içeriği bul
    lines = clean_context.split('\n')
    meaningful_lines = []
    
    for line in lines:
        line = line.strip()
        # Anlamsız satırları atla
        if len(line) < 10 or line.startswith('http') or 'badge' in line.lower():
            continue
        if len(line.split()) > 3:  # En az 3 kelime içeren satırlar
            meaningful_lines.append(line)
    
    meaningful_text = ' '.join(meaningful_lines[:5])  # İlk 5 anlamlı satır
    
    if not meaningful_text:
        meaningful_text = clean_context[:500]  # Fallback
    
    question_lower = question.lower()
    
    if "nedir" in question_lower or "ne demek" in question_lower:
        return f"{meaningful_text[:400]}... [Devamı için detaylı sonuçlara bakın]"
    
    elif "nasıl" in question_lower or "yapılır" in question_lower or "kurulur" in question_lower:
        return f"{meaningful_text[:450]}... [Adımlar için detaylı sonuçları inceleyin]"
    
    elif "ne işe yarar" in question_lower or "kullanım" in question_lower:
        return f"{meaningful_text[:400]}... [Kullanım detayları için yukarıdaki sonuçlara bakın]"
    
    elif "bileşen" in question_lower or "component" in question_lower:
        return f"{meaningful_text[:380]}... [Bileşen listesi için detaylı sonuçları görün]"
    
    elif "rag" in question_lower:
        return f"{meaningful_text[:350]}... [RAG mimarisi detayları için sonuçlara bakın]"
    
    else:
        return f"{meaningful_text[:500]}... [Detaylı bilgi için yukarıdaki sonuçları okuyun]"

# --- 4. Kullanıcı Sorusunu Yanıtlama ---
def user_input(user_question):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        if not os.path.exists("faiss_index"):
            st.error("FAISS index bulunamadı. Lütfen önce index oluşturun.")
            return
            
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question, k=3)
        
        st.success(f"✅ '{user_question}' sorusu için {len(docs)} sonuç bulundu:")
        
        # Tüm dokümanları birleştir ve temizle
        all_context = "\n".join([doc.page_content for doc in docs])
        
        # Temiz yanıt oluştur
        with st.spinner("Yanıt oluşturuluyor..."):
            response = generate_response(all_context, user_question)
            
            st.info("💡 **Yanıt:**")
            st.write(response)
        
        # Detaylı ve TEMİZ sonuçlar
        st.divider()
        st.subheader("📄 Detaylı Sonuçlar")
        
        for i, doc in enumerate(docs):
            clean_content = clean_html_tags(doc.page_content)
            # Çok kısa içerikleri filtrele
            if len(clean_content) > 50:
                with st.expander(f"📖 Sonuç {i+1}"):
                    st.write(clean_content)
        
    except Exception as e:
        st.error(f"Hata: {str(e)}")

# --- STREAMLIT ARAYÜZÜ ---
st.set_page_config(
    page_title="Akbank GenAI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.header("🤖 Akbank GenAI Bootcamp Projesi: RAG Chatbot")

# Session state
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""

# Sidebar
with st.sidebar:
    st.subheader("📋 Proje Bilgisi")
    st.write("RAG (Retrieval Augmented Generation) tabanlı chatbot")
    st.write("**Veri:** LangChain dokümantasyonu")
    
    st.divider()
    
    st.subheader("⚙️ Sistem")
    
    if st.button("🔄 Index'i Temizle ve Yeniden Oluştur", use_container_width=True):
        if os.path.exists("faiss_index"):
            import shutil
            try:
                shutil.rmtree("faiss_index")
                st.success("✅ Eski index temizlendi!")
            except Exception as e:
                st.error(f"Silme hatası: {str(e)}")
        
        with st.spinner("Temiz index oluşturuluyor..."):
            try:
                text_chunks = get_text_chunks("data.txt")
                if text_chunks:
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.success("✅ Temiz index oluşturuldu!")
                    else:
                        st.error("❌ Index oluşturulamadı!")
                else:
                    st.error("❌ Veri yüklenemedi!")
            except Exception as e:
                st.error(f"Hata: {str(e)}")

# Ana içerik
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Sohbet")
    
    if os.path.exists("faiss_index"):
        st.success("✅ Sistem hazır! Sorularınızı sorabilirsiniz.")
    else:
        st.warning("⚠️ Lütfen önce sidebar'dan index oluşturun!")
    
    st.divider()
    
    user_question = st.text_input(
        "LangChain hakkında sorunuzu girin:",
        placeholder="Örnek: LangChain nedir? Nasıl kullanılır?",
        key="question_input"
    )
    
    if st.session_state.current_question:
        user_question = st.session_state.current_question
        st.session_state.current_question = ""
    
    if user_question:
        user_input(user_question)

with col2:
    st.subheader("🚀 Örnek Sorular")
    
    examples = [
        "LangChain nedir?",
        "LangChain nasıl kurulur?",
        "LangChain ne işe yarar?",
        "LangChain'in temel bileşenleri nelerdir?",
        "RAG (Retrieval Augmented Generation) nedir?",
        "LangChain ile neler yapılabilir?"
    ]
    
    for example in examples:
        if st.button(example, key=example, use_container_width=True):
            st.session_state.current_question = example
            st.rerun()

# Footer
st.divider()
st.caption("Akbank GenAI Bootcamp - RAG Chatbot | Temizlenmiş İçerik")

# Debug
with st.expander("🔧 Sistem Durumu"):
    st.write(f"FAISS Index: {'✅ Var' if os.path.exists('faiss_index') else '❌ Yok'}")
    st.write(f"Data.txt: {'✅ Var' if os.path.exists('data.txt') else '❌ Yok'}")