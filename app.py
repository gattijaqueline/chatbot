import os
import streamlit as st
from pypdf import PdfReader
from openai import AzureOpenAI

# ==============================================================================
# 1. CONFIGURAÇÕES DA PÁGINA E CSS (INTERFACE COM BORDA ESTILIZADA)
# ==============================================================================
st.set_page_config(
    page_title="ChatBot com RAG | Azure AI Foundry",
    page_icon="🤖",
    layout="wide"
)

# Injeção de CSS personalizado para criar uma área de chat com borda e acabamento moderno
custom_css = """
<style>
/* Borda e moldura elegante ao redor da área principal do chat */
.main-chat-container {
    border: 2px solid #3b82f6;
    border-radius: 14px;
    padding: 24px;
    background-color: rgba(255, 255, 255, 0.02);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    margin-bottom: 25px;
    min-height: 480px;
}

/* Cabeçalho centralizado */
.app-header {
    text-align: center;
    margin-bottom: 20px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==============================================================================
# 2. CONFIGURAÇÃO DA API (MICROSOFT AZURE AI FOUNDRY)
# ==============================================================================
API_KEY = "6t2pTNsmNafDyo53MDAFIrvvQRBZXP7dLcr5j3TLxMS34LQVq3ZUJQQJ99CHACHYHv6XJ3w3AAAAACOGT7os"
DEPLOYMENT_NAME = "gpt-4.1-mini"
ENDPOINT = "https://marcelomaffeis-05082026-resource.services.ai.azure.com/"
API_VERSION = "2024-02-15-preview"

# Inicialização em cache do cliente oficial AzureOpenAI
@st.cache_resource
def get_azure_client():
    return AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION
    )

client = get_azure_client()

# ==============================================================================
# 3. EXTRAÇÃO DE TEXTO PARA RAG (PDF E TXT)
# ==============================================================================
def extract_text_from_file(uploaded_file):
    """Extrai e consolida o texto de arquivos .pdf ou .txt."""
    text = ""
    try:
        if uploaded_file.name.lower().endswith(".txt"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.name.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.sidebar.error(f"Erro ao ler o documento: {e}")
    return text.strip()

# ==============================================================================
# 4. GESTÃO DE ESTADO (SESSION STATE)
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou seu assistente inteligente. Carregue um PDF ou TXT na barra lateral para tirar dúvidas com RAG ou pergunte diretamente!"}
    ]

if "document_context" not in st.session_state:
    st.session_state.document_context = ""

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# ==============================================================================
# 5. BARRA LATERAL: UPLOAD E CONTROLES
# ==============================================================================
with st.sidebar:
    st.header("📂 Base de Conhecimento (RAG)")
    st.caption("Envie arquivos para enriquecer o contexto das respostas.")

    uploaded_file = st.file_uploader(
        "Carregue um arquivo (.pdf ou .txt)",
        type=["pdf", "txt"],
        help="O conteúdo textual será extraído e injetado nas chamadas ao modelo."
    )

    if uploaded_file is not None:
        if st.session_state.current_file_name != uploaded_file.name:
            with st.spinner("Processando e indexando arquivo..."):
                extracted_content = extract_text_from_file(uploaded_file)
                st.session_state.document_context = extracted_content
                st.session_state.current_file_name = uploaded_file.name
            st.success(f"Carregado: {uploaded_file.name}")

        st.markdown(f"**Documento ativo:** `{st.session_state.current_file_name}`")
        st.markdown(f"**Caracteres indexados:** `{len(st.session_state.document_context)}`")

        with st.expander("Prévia do texto extraído"):
            preview_len = min(1000, len(st.session_state.document_context))
            st.text(st.session_state.document_context[:preview_len] + ("..." if len(st.session_state.document_context) > 1000 else ""))
    else:
        st.session_state.document_context = ""
        st.session_state.current_file_name = None
        st.info("Nenhum arquivo enviado. O bot responderá utilizando sua base geral.")

    st.divider()
    if st.button("🗑️ Limpar Histórico", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Histórico limpo. Como posso ajudar agora?"}
        ]
        st.rerun()

# ==============================================================================
# 6. INTERFACE PRINCIPAL E HISTÓRICO DE MENSAGENS
# ==============================================================================
st.markdown("<div class='app-header'><h1>🤖 ChatBot RAG com Microsoft Azure AI</h1><p>Assistente com extração documental em tempo real</p></div>", unsafe_allow_html=True)

# Div com a borda CSS estilizada
st.markdown("<div class='main-chat-container'>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 7. ENTRADA DE DADOS E LÓGICA RAG COM STREAMING
# ==============================================================================
user_prompt = st.chat_input("Digite sua dúvida sobre o documento ou outro assunto...")

if user_prompt:
    # Registra e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Constrói o System Prompt com o contexto do RAG
    system_instruction = (
        "Você é um assistente analítico, claro e objetivo. "
        "Se houver contexto de um documento anexado, use-o prioritariamente para fundamentar sua resposta. "
        "Caso a informação não esteja presente no texto, informe que o documento não cita o assunto antes de complementar."
    )

    if st.session_state.document_context:
        # Segmentação preventiva de contexto para otimização de tokens
        context_slice = st.session_state.document_context[:25000]
        system_instruction += f"\n\n[DOCUMENTO CARREGADO]:\n{context_slice}"

    # Montagem do histórico para o modelo
    messages_payload = [{"role": "system", "content": system_instruction}]
    for msg in st.session_state.messages:
        messages_payload.append({"role": msg["role"], "content": msg["content"]})

    # Chamada com streaming para melhor experiência de usuário
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages_payload,
                temperature=0.3,
                stream=True
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Ocorreu um erro ao comunicar com a Azure OpenAI: {e}"
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
