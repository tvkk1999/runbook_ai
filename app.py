# app.py

import streamlit as st
import tempfile
import os
from pathlib import Path
from PIL import Image
import pandas as pd

from document_processor import DocumentProcessor
from guardrails import GuardrailsManager
from llm_manager import LocalLLMManager
from vector_store import VectorStore


class RunbookApp:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.guardrails = GuardrailsManager()
        self.llm_manager = LocalLLMManager()
        self.vector_store = VectorStore()

        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def setup_sidebar(self):
        st.sidebar.title("⚙️ Configuration")
        available_models = ["llama3.1:8b", "mistral:7b", "codellama:7b"]
        selected_model = st.sidebar.selectbox("Select LLM Model", available_models, index=0)
        if selected_model != self.llm_manager.model_name:
            self.llm_manager.model_name = selected_model

        st.sidebar.subheader("🔧 System Status")
        ollama_status = self.llm_manager.ensure_ollama_running()
        st.sidebar.write(f"Ollama: {'🟢 Running' if ollama_status else '🔴 Stopped'}")
        if not ollama_status and st.sidebar.button("Start Ollama"):
            st.sidebar.info("Please start Ollama manually")

        st.sidebar.subheader("📚 Document Management")
        if st.sidebar.button("Clear Documents"):
            self.vector_store.clear_collection()
            st.session_state.documents_loaded = False
            st.sidebar.success("Documents cleared!")

    def document_upload_section(self):
        st.header("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Runbook Documents",
            type=['pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload PDF or Word documents containing your runbooks"
        )

        if uploaded_files and st.button("Process Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            all_chunks = []

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                try:
                    file_type = Path(uploaded_file.name).suffix[1:].lower()
                    chunks_with_meta = self.doc_processor.process_document(tmp_path, file_type)
                    
                    def clean_metadata(md):
                        # Remove any 'content' key
                        md = dict(md)
                        md.pop("content", None)
                        # For safety, cast any list/dict/other values to string
                        for k,v in md.items():
                            if not isinstance(v, (str, int, float, bool, type(None))):
                                md[k] = str(v)
                        return md

                    for chunk in chunks_with_meta:
                        content = chunk.get("content", "")
                        metadata = clean_metadata(chunk.get("metadata", {}))
                        metadata.update({
                            "source": uploaded_file.name,
                            "file_type": file_type
                        })
                        metadata["chunk_type"] = chunk.get("type", "text")

                        all_chunks.append({
                            "type": chunk.get("type", "text"),
                            "content": content,
                            "metadata": metadata
                        })
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    os.unlink(tmp_path)

                progress_bar.progress((i + 1) / len(uploaded_files))

            if all_chunks:
                self.vector_store.add_documents(all_chunks)
                st.session_state.documents_loaded = True
                st.success(f"Successfully processed {len(uploaded_files)} documents with {len(all_chunks)} chunks!")

            progress_bar.empty()
            status_text.empty()

    def display_chunk(self, chunk):
        """Display text, image, or table chunk appropriately."""
        chunk_type = chunk.get("type", "text")
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})

        if chunk_type == "image":
            image_path = metadata.get("file")
            caption = metadata.get("caption", "Image")
            if image_path and os.path.exists(image_path):
                img = Image.open(image_path)
                st.image(img, caption=caption)
            else:
                st.write(f"[Image not found: {caption}]")
        elif chunk_type == "table":
            table_content = content
            if isinstance(table_content, list) and all(isinstance(row, list) for row in table_content):
                try:
                    df = pd.DataFrame(table_content)
                    st.table(df)
                except Exception:
                    st.write(table_content)
            else:
                st.write(content)
        else:
            st.write(content)

    def chat_section(self):
        st.header("💬 Ask Questions")

        if not st.session_state.documents_loaded:
            st.warning("Please upload and process documents first.")
            return

        # Show previous chat history
        for i, (question, answer_chunks) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**Q{i + 1}:** {question}")
                for chunk in answer_chunks:
                    self.display_chunk(chunk)
                st.divider()

        user_query = st.text_input(
            "Enter your question:",
            placeholder="How do I restart the service?",
            key="user_query"
        )

        if st.button("Ask Question") and user_query:
            with st.spinner("Processing your question..."):
                try:
                    validated_query = self.guardrails.validate_input(
                        user_query,
                        {"documents": ["available"]}
                    )
                    
                    print(validated_query)

                    results = self.vector_store.search_similar(validated_query, n_results=5)
                    
                    print(results)

                    docs = results.get("documents", [])
                    metas = results.get("metadatas", [])

                    if not docs or not metas:
                        st.error("No relevant information found in documents.")
                        return

                    retrieved_chunks = []
                    for content, metadata in zip(docs[0], metas[0]):
                        chunk_type = metadata.get("chunk_type", "text")
                        chunk = {
                            "type": chunk_type,
                            "content": content,
                            "metadata": metadata
                        }
                        retrieved_chunks.append(chunk)

                    # Build context string for LLM prompt (text + table captions/content)
                    context_texts = []
                    for chunk in retrieved_chunks:
                        ctype = chunk["type"]
                        if ctype == "text":
                            context_texts.append(chunk["content"])
                        elif ctype == "table":
                            caption = chunk["metadata"].get("caption", "")
                            table_content = chunk["content"]
                            if isinstance(table_content, list) and all(isinstance(row, list) for row in table_content):
                                md_table = "\n".join([" | ".join(row) for row in table_content])
                                context_texts.append(f"{caption}\n{md_table}")
                            else:
                                context_texts.append(caption)
                        elif ctype == "image":
                            caption = chunk["metadata"].get("caption", "")
                            context_texts.append(f"{caption} [Image omitted in prompt]")

                    context_for_llm = "\n\n".join(context_texts)

                    # Generate answer using LLM
                    response = self.llm_manager.generate_response(validated_query, context_for_llm)
                    
                    print()
                    print()
                    print("res=>  ",response)

                    # Validate output
                    if self.guardrails.validate_output(response, context_texts):
                        st.success("Response generated successfully!")
                        st.markdown(f"**Answer:** {response}")

                        with st.expander("📚 Source References and Evidence"):
                            for chunk in retrieved_chunks:
                                self.display_chunk(chunk)

                        answer_text_chunk = {
                            "type": "text",
                            "content": response,
                            "metadata": {"source": "LLM Answer"}
                        }
                        st.session_state.chat_history.append((user_query, [answer_text_chunk] + retrieved_chunks))
                    else:
                        st.error("Response failed validation checks.")

                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

    def run(self):
        st.set_page_config(page_title="Runbook AI Assistant", page_icon="📖", layout="wide")

        st.title("📖 Runbook AI Assistant")
        st.markdown("Upload your runbook documents and ask questions!")

        self.setup_sidebar()

        col1, col2 = st.columns([1, 1])
        with col1:
            self.document_upload_section()
        with col2:
            self.chat_section()


if __name__ == "__main__":
    app = RunbookApp()
    app.run()
