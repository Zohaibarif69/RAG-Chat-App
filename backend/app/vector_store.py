from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import PyPDF2
from io import BytesIO
from typing import List, Dict, Tuple
import os
from rank_bm25 import BM25Okapi
import numpy as np

# Lazy-initialized globals
_embedding = None
_vectorstore = None
_llm = None

def reset_vectorstore():
    """Reset the vectorstore (for clearing database)"""
    global _vectorstore
    _vectorstore = None

def get_embedding():
    """Initialize and return embeddings (lazy initialization)"""
    global _embedding
    if _embedding is None:
        print("Initializing HuggingFace embeddings model...")
        try:
            _embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("Embeddings model loaded successfully!")
        except Exception as e:
            print(f"ERROR initializing embeddings: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _embedding

def get_chroma_vectorstore():
    """Initialize and return vectorstore (lazy initialization)"""
    global _vectorstore
    if _vectorstore is None:
        embedding = get_embedding()
        # Get absolute path for chroma_db directory
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        chroma_path = os.path.abspath(chroma_path)
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        _vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=embedding,
            persist_directory=chroma_path
        )
    return _vectorstore

def get_llm():
    """Initialize and return LLM (lazy initialization)"""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
    return _llm


def get_vectorstore():
    """Return the vectorstore"""
    return get_chroma_vectorstore()


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    text = ""
    metadata = {"source": "pdf_document"}
    
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text, metadata


def chunk_documents(text: str, metadata: Dict) -> List[Document]:
    """Split documents into chunks with overlap"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    # Create Document objects with metadata
    documents = [
        Document(page_content=chunk, metadata={**metadata, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    
    return documents


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize scores to 0-1 range using min-max normalization"""
    if not scores or len(scores) == 0:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if min_score == max_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_retrieve(query: str, k: int = 5, alpha: float = 0.6) -> Tuple[List[Document], List[Dict]]:
    """
    Hybrid retrieval combining BM25 (keyword) and vector similarity search.
    
    Args:
        query: Search query
        k: Number of results to return
        alpha: Weight for vector similarity (0-1). Result = alpha*vec + (1-alpha)*bm25
               alpha=0.6 gives 60% to semantic, 40% to keyword matching
    
    Returns:
        Tuple of (documents, sources with relevance scores)
    """
    try:
        vectorstore = get_chroma_vectorstore()
        collection = vectorstore._collection
        
        # Get all documents from collection
        all_data = collection.get()
        if not all_data or not all_data.get('ids'):
            return [], []
        
        all_docs = all_data.get('documents', [])
        all_ids = all_data.get('ids', [])
        all_metadatas = all_data.get('metadatas', [])
        
        # --- VECTOR SIMILARITY SEARCH ---
        # Get vector similarity scores (retrieve from all docs, map by content)
        vector_results = vectorstore.similarity_search_with_score(query, k=min(k*3, len(all_docs)))
        
        # Create mapping of document content to vector scores
        vector_content_scores = {}
        for doc, score in vector_results:
            content = doc.page_content
            if content not in vector_content_scores:  # Keep highest score
                vector_content_scores[content] = float(score)
        
        # --- BM25 KEYWORD SEARCH ---
        # Tokenize all documents
        tokenized_docs = [doc.lower().split() for doc in all_docs]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores for query
        query_tokens = query.lower().split()
        bm25_scores_raw = bm25.get_scores(query_tokens)
        
        # Create BM25 mapping by document index
        bm25_scores_dict = {i: float(score) for i, score in enumerate(bm25_scores_raw)}
        
        # --- NORMALIZE BOTH SCORE SETS ---
        # Get all vector scores for normalization
        all_vector_scores = []
        for i, doc_content in enumerate(all_docs):
            if doc_content in vector_content_scores:
                all_vector_scores.append(vector_content_scores[doc_content])
        
        # Normalize vector scores
        normalized_vector_dict = {}
        if all_vector_scores:
            norm_vector_scores = normalize_scores(all_vector_scores)
            score_idx = 0
            for i, doc_content in enumerate(all_docs):
                if doc_content in vector_content_scores:
                    normalized_vector_dict[i] = norm_vector_scores[score_idx]
                    score_idx += 1
                else:
                    normalized_vector_dict[i] = 0.0
        else:
            normalized_vector_dict = {i: 0.0 for i in range(len(all_docs))}
        
        # Normalize BM25 scores
        bm25_values = list(bm25_scores_dict.values())
        if bm25_values and max(bm25_values) > 0:
            normalized_bm25 = normalize_scores(bm25_values)
            normalized_bm25_dict = {i: normalized_bm25[i] for i in range(len(all_docs))}
        else:
            normalized_bm25_dict = {i: 0.0 for i in range(len(all_docs))}
        
        # --- MERGE RESULTS WITH WEIGHTED COMBINATION ---
        # Combined score = (alpha * vector) + ((1-alpha) * bm25)
        combined_scores = {}
        for idx in range(len(all_docs)):
            vec_score = normalized_vector_dict.get(idx, 0.0)
            bm25_score = normalized_bm25_dict.get(idx, 0.0)
            combined = (alpha * vec_score) + ((1 - alpha) * bm25_score)
            combined_scores[idx] = {
                'combined': combined,
                'vector': vec_score,
                'bm25': bm25_score
            }
        
        # Sort by combined score and get top k
        sorted_indices = sorted(combined_scores.keys(), 
                               key=lambda x: combined_scores[x]['combined'], 
                               reverse=True)[:k]
        
        # Build final results
        documents = []
        sources = []
        for idx in sorted_indices:
            doc_content = all_docs[idx]
            metadata = all_metadatas[idx]
            doc = Document(page_content=doc_content, metadata=metadata)
            documents.append(doc)
            
            scores = combined_scores[idx]
            sources.append({
                "source": metadata.get("source", "Unknown"),
                "chunk": metadata.get("chunk", 0),
                "relevance": round(scores['combined'], 3),
                "vector_score": round(scores['vector'], 3),
                "bm25_score": round(scores['bm25'], 3),
                "method": "hybrid"
            })
        
        return documents, sources
    
    except Exception as e:
        print(f"Error in hybrid_retrieve: {e}")
        import traceback
        traceback.print_exc()
        return retrieve_context_vector_only(query, k)


def retrieve_context_vector_only(query: str, k: int = 5) -> Tuple[List[Document], List[Dict]]:
    """Pure vector similarity search (fallback)"""
    results = get_chroma_vectorstore().similarity_search_with_score(query, k=k)
    
    documents = []
    sources = []
    
    for doc, score in results:
        documents.append(doc)
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "chunk": doc.metadata.get("chunk", 0),
            "relevance": float(score),
            "method": "vector_only"
        })
    
    return documents, sources


def add_documents(pdf_bytes: bytes, filename: str) -> Dict:
    """Process and add PDF documents to vector store"""
    try:
        print(f"\n=== UPLOAD DEBUG: Starting add_documents for {filename} ===", flush=True)
        print(f"DEBUG: PDF size: {len(pdf_bytes)} bytes", flush=True)
        
        # Extract text from PDF
        text, metadata = extract_text_from_pdf(pdf_bytes)
        print(f"DEBUG: Extracted text length: {len(text)} characters", flush=True)
        print(f"DEBUG: Text preview: {text[:200]}...", flush=True)
        
        # Check if text is empty
        if not text or not text.strip():
            print(f"DEBUG: PDF is empty!", flush=True)
            return {
                "success": False,
                "error": "PDF contains no extractable text. Make sure it's a valid text-based PDF (not scanned images)."
            }
        
        metadata["source"] = filename
        print(f"DEBUG: Set source to: {filename}", flush=True)
        
        # Chunk documents
        documents = chunk_documents(text, metadata)
        print(f"DEBUG: Created {len(documents)} chunks", flush=True)
        
        # Check if chunks are empty or too small
        if not documents:
            print(f"DEBUG: No chunks created!", flush=True)
            return {
                "success": False,
                "error": "Could not create chunks from PDF text"
            }
        
        # Filter out empty chunks
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        print(f"DEBUG: After filtering: {len(valid_documents)} valid documents", flush=True)
        
        if not valid_documents:
            print(f"DEBUG: No valid documents after filtering!", flush=True)
            return {
                "success": False,
                "error": "All chunks are empty or contain only whitespace"
            }
        
        print(f"DEBUG: Adding {len(valid_documents)} documents to vectorstore", flush=True)
        
        # Add to vectorstore with error handling
        try:
            print(f"DEBUG: Getting vectorstore instance...", flush=True)
            vectorstore = get_chroma_vectorstore()
            print(f"DEBUG: Got vectorstore, now adding documents...", flush=True)
            vectorstore.add_documents(valid_documents)
            print(f"DEBUG: Successfully added documents to vectorstore", flush=True)
            
            # Verify the data was added
            collection = vectorstore._collection
            all_data = collection.get()
            total_ids = len(all_data.get('ids', []))
            print(f"DEBUG: Total items now in collection: {total_ids}", flush=True)
            print(f"DEBUG: Collection metadata: {all_data.get('metadatas', [])[:3]}", flush=True)
            
        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: Error adding documents: {error_msg}", flush=True)
            import traceback
            traceback.print_exc()
            if "empty" in error_msg.lower() or "embeddings" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Failed to create embeddings for PDF content: {error_msg}"
                }
            raise
        
        print(f"=== UPLOAD DEBUG: Successfully completed upload ===", flush=True)
        return {
            "success": True,
            "message": f"Successfully added {len(valid_documents)} chunks from {filename}",
            "chunks_count": len(valid_documents)
        }
    except Exception as e:
        print(f"\n=== ERROR in add_documents: {e} ===", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def retrieve_context(query: str, k: int = 5, use_hybrid: bool = True, filters: Dict = None, alpha: float = 0.6) -> tuple[List[Document], List[Dict]]:
    """
    Retrieve top k relevant documents using hybrid search with optional metadata filtering.
    
    Args:
        query: Search query
        k: Number of results to return
        use_hybrid: Use hybrid retrieval (BM25 + vector) vs vector-only
        filters: Optional metadata filters (e.g., {"year": "2024", "category": "finance"})
        alpha: Weight for vector similarity (0.6 = 60% vector, 40% BM25)
    
    Returns:
        Tuple of (documents, sources with relevance scores)
    """
    if use_hybrid:
        return hybrid_retrieve(query, k, alpha)
    else:
        return retrieve_context_vector_only(query, k)


def rewrite_query(original_query: str, num_variations: int = 4) -> List[str]:
    """
    Use Groq to rewrite/expand the original query into multiple variations.
    This helps handle vague or ambiguous queries by generating alternative phrasings.
    
    Args:
        original_query: The original user query
        num_variations: Number of query variations to generate (default 4)
    
    Returns:
        List of query variations including the original
    """
    try:
        llm = get_llm()
        
        prompt_template = PromptTemplate(
            input_variables=["question", "num"],
            template="""You are an expert at reformulating questions for better information retrieval.

Original question: {question}

Your task: Generate {num} different ways to ask the same question, each emphasizing different aspects.
These will be used for multiple search retrieval to find the best information.

Requirements:
- Each variation should be a complete question
- Variations should emphasize different keywords and phrasings
- Include synonyms and alternative terms
- Make them search-engine friendly

Output format: Return ONLY the questions, one per line, numbered 1-{num}.
Do not include explanations or preamble.

Generate {num} question variations:"""
        )
        
        prompt = prompt_template.format(question=original_query, num=num_variations)
        response = llm.invoke(prompt)
        
        # Parse the response to extract questions
        response_text = response.content
        lines = response_text.strip().split('\n')
        
        # Extract questions (remove numbering if present)
        variations = []
        for line in lines:
            cleaned = line.strip()
            # Remove numbering patterns like "1.", "1)", "1 - ", etc.
            if cleaned:
                # Remove leading numbers and common separators
                import re
                cleaned = re.sub(r'^[\d]+[\.\)\-\s]+', '', cleaned).strip()
                if cleaned and len(cleaned) > 3:  # Filter out empty or too-short results
                    variations.append(cleaned)
        
        # Always include original query
        if original_query not in variations:
            variations.insert(0, original_query)
        
        # Limit to desired number
        variations = variations[:num_variations]
        
        print(f"Generated {len(variations)} query variations:")
        for i, var in enumerate(variations, 1):
            print(f"  {i}. {var}")
        
        return variations
    
    except Exception as e:
        print(f"Error in query rewriting: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original query
        return [original_query]


def multi_query_retrieve(query: str, k: int = 5, use_query_rewriting: bool = True) -> tuple[List[Document], List[Dict]]:
    """
    Retrieve documents using multiple query variations for better coverage.
    
    Args:
        query: Original user query
        k: Number of top results to return
        use_query_rewriting: Whether to use query rewriting (default True)
    
    Returns:
        Tuple of (merged documents, merged sources with retrieval metadata)
    """
    try:
        # Generate query variations
        if use_query_rewriting:
            query_variations = rewrite_query(query, num_variations=4)
        else:
            query_variations = [query]
        
        print(f"\nRetrieving with {len(query_variations)} query variations...")
        
        # Retrieve for each variation
        all_results = {}  # key: document content, value: metadata
        all_sources = {}  # key: document content, value: source info
        
        for i, variant_query in enumerate(query_variations, 1):
            print(f"\n  Query {i}: {variant_query}")
            
            # Retrieve using hybrid search
            docs, sources = hybrid_retrieve(variant_query, k=min(k * 2, 10), alpha=0.6)
            
            print(f"    ↳ Found {len(docs)} documents")
            
            # Merge results
            for doc, source in zip(docs, sources):
                doc_content = doc.page_content
                
                if doc_content not in all_results:
                    # First time seeing this document
                    all_results[doc_content] = doc
                    all_sources[doc_content] = {
                        "source": source.get("source"),
                        "chunk": source.get("chunk"),
                        "relevance_scores": [source.get("relevance", 0.0)],
                        "vector_scores": [source.get("vector_score", 0.0)],
                        "bm25_scores": [source.get("bm25_score", 0.0)],
                        "query_variants_found": [i],
                        "source_data": source
                    }
                else:
                    # Seen this document before - aggregate scores
                    all_sources[doc_content]["relevance_scores"].append(source.get("relevance", 0.0))
                    all_sources[doc_content]["vector_scores"].append(source.get("vector_score", 0.0))
                    all_sources[doc_content]["bm25_scores"].append(source.get("bm25_score", 0.0))
                    all_sources[doc_content]["query_variants_found"].append(i)
        
        # Aggregate and rank results
        ranked_results = []
        for doc_content, source_info in all_sources.items():
            # Calculate average scores
            avg_relevance = sum(source_info["relevance_scores"]) / len(source_info["relevance_scores"])
            avg_vector = sum(source_info["vector_scores"]) / len(source_info["vector_scores"])
            avg_bm25 = sum(source_info["bm25_scores"]) / len(source_info["bm25_scores"])
            
            # Boost score based on how many query variants found this result
            variant_boost = len(source_info["query_variants_found"]) / len(query_variations)
            boosted_relevance = (avg_relevance * 0.7) + (variant_boost * 0.3)
            
            ranked_results.append({
                "content": doc_content,
                "relevance": round(boosted_relevance, 3),
                "vector_score": round(avg_vector, 3),
                "bm25_score": round(avg_bm25, 3),
                "found_in_variants": source_info["query_variants_found"],
                "source": source_info["source"],
                "chunk": source_info["chunk"]
            })
        
        # Sort by boosted relevance
        ranked_results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Return top k
        final_results = ranked_results[:k]
        
        # Build final documents and sources
        final_docs = []
        final_sources = []
        
        for result in final_results:
            doc = all_results[result["content"]]
            final_docs.append(doc)
            final_sources.append({
                "source": result["source"],
                "chunk": result["chunk"],
                "relevance": result["relevance"],
                "vector_score": result["vector_score"],
                "bm25_score": result["bm25_score"],
                "found_in_query_variants": result["found_in_variants"],
                "method": "multi_query_hybrid"
            })
        
        print(f"\n✓ Merged {len(all_sources)} unique documents, returning top {len(final_docs)}")
        
        return final_docs, final_sources
    
    except Exception as e:
        print(f"Error in multi_query_retrieve: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to simple retrieval
        return hybrid_retrieve(query, k)


def generate_answer(query: str, context_docs: List[Document]) -> tuple[str, List[Dict]]:
    """Generate answer using Groq LLM with retrieved context"""
    
    # Prepare context
    context_text = "\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in context_docs
    ])
    
    # Create prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the following context to answer the question.
        
Context:
{context}

Question: {question}

Answer: """
    )
    
    # Generate answer
    prompt = prompt_template.format(context=context_text, question=query)
    answer = get_llm().invoke(prompt)
    
    return answer.content


def rag_query(query: str, k: int = 5, use_query_rewriting: bool = True) -> Dict:
    """
    Complete RAG pipeline with optional query rewriting for smarter retrieval.
    
    Args:
        query: User's question
        k: Number of documents to retrieve
        use_query_rewriting: Enable query rewriting for better coverage (default True)
    
    Returns:
        Dictionary with success status, answer, and sources
    """
    try:
        # Retrieve relevant documents (with optional query rewriting)
        if use_query_rewriting:
            context_docs, sources = multi_query_retrieve(query, k)
        else:
            context_docs, sources = hybrid_retrieve(query, k)
        
        if not context_docs:
            return {
                "success": False,
                "error": "No relevant documents found"
            }
        
        # Generate answer
        answer = generate_answer(query, context_docs)
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"Error in rag_query: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def get_all_documents() -> List[Dict]:
    """Get list of all documents in the database"""
    try:
        print(f"\n=== DEBUG /documents endpoint ===", flush=True)
        vectorstore = get_chroma_vectorstore()
        # Get all data from the collection
        collection = vectorstore._collection
        data = collection.get()
        
        print(f"DEBUG: Collection.get() returned:", flush=True)
        print(f"  - IDs: {len(data.get('ids', []))} items", flush=True)
        print(f"  - Metadatas: {len(data.get('metadatas', []))} items", flush=True)
        print(f"  - Documents: {len(data.get('documents', []))} items", flush=True)
        
        # Extract unique sources
        if data and data.get("metadatas") and len(data["metadatas"]) > 0:
            sources_set = set()
            chunk_counts = {}
            print(f"DEBUG: Processing {len(data['metadatas'])} metadata entries:", flush=True)
            for i, metadata in enumerate(data["metadatas"]):
                source = metadata.get("source", "Unknown")
                print(f"  [{i}] source={source}", flush=True)
                sources_set.add(source)
                chunk_counts[source] = chunk_counts.get(source, 0) + 1
            
            documents = [
                {"name": source, "chunks": chunk_counts[source]}
                for source in sorted(sources_set)
            ]
            print(f"DEBUG: Returning {len(documents)} documents: {documents}", flush=True)
            print(f"=== DEBUG /documents endpoint DONE ===", flush=True)
            return documents
        print(f"DEBUG: No metadata found in collection - DATABASE IS EMPTY!", flush=True)
        print(f"=== DEBUG /documents endpoint DONE ===", flush=True)
        return []
    except Exception as e:
        print(f"\n=== Error getting documents: {e} ===", flush=True)
        import traceback
        traceback.print_exc()
        print(f"=== Error getting documents DONE ===", flush=True)
        return []


def delete_document(source_name: str) -> Dict:
    """Delete all chunks of a specific document by source name"""
    try:
        vectorstore = get_chroma_vectorstore()
        collection = vectorstore._collection
        
        # Get all data
        data = collection.get()
        if not data or not data.get("ids"):
            return {"success": False, "error": "No documents found"}
        
        # Find IDs that match the source
        ids_to_delete = []
        for i, metadata in enumerate(data.get("metadatas", [])):
            if metadata.get("source") == source_name:
                ids_to_delete.append(data["ids"][i])
        
        if not ids_to_delete:
            return {"success": False, "error": f"Document '{source_name}' not found"}
        
        print(f"Deleting {len(ids_to_delete)} chunks for document: {source_name}")
        
        # Delete the documents
        try:
            collection.delete(ids=ids_to_delete)
            print(f"Successfully deleted {len(ids_to_delete)} chunks")
        except OSError as e:
            print(f"File access error during delete: {e}")
            # Try alternative deletion approach
            collection._client.delete(ids=ids_to_delete)
        
        return {
            "success": True,
            "message": f"Deleted {len(ids_to_delete)} chunks from {source_name}"
        }
    except Exception as e:
        print(f"Error deleting document: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Failed to delete: {str(e)}"
        }
