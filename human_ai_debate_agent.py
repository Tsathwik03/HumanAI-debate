from __future__ import annotations
import os
import sys
import json
import argparse
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Use environment variable for API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_groq import ChatGroq
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db")

DEBATE_AGENT_INSTRUCTIONS = (
    "You are a rigorous debate agent. Produce a concise rebuttal to the opponent's argument, "
    "then state your own claim and 2-4 supporting numbered points. When you reference facts "
    "present in the EVIDENCE block, cite them with bracketed numbers like [1], [2]. "
    "Do not invent facts beyond the EVIDENCE."
)

@dataclass
class DebateTurn:
    actor: str
    text: str
    citations: List[Dict[str, Any]]

@dataclass
class DebateTranscript:
    topic: str
    turns: List[DebateTurn]

def load_documents(path: str) -> List:
    print(f"[DEBUG] Loading documents from: {path}")
    if not os.path.isdir(path):
        print(f"❌ Directory not found: {path}")
        raise FileNotFoundError(f"Ingest path not found: {path}")
    
    docs = []
    file_count = 0
    
    for root, _, files in os.walk(path):
        print(f"[DEBUG] Checking directory: {root}")
        print(f"[DEBUG] Files found: {files}")
        
        for f in files:
            full = os.path.join(root, f)
            try:
                if f.lower().endswith(".txt"):
                    print(f"[DEBUG] Loading TXT file: {f}")
                    docs.extend(TextLoader(full, encoding="utf-8").load())
                    file_count += 1
                elif f.lower().endswith(".pdf"):
                    print(f"[DEBUG] Loading PDF file: {f}")
                    docs.extend(PyPDFLoader(full).load())
                    file_count += 1
            except Exception as e:
                print(f"❌ Error loading {f}: {e}")
    
    print(f"[DEBUG] Total files processed: {file_count}")
    print(f"[DEBUG] Total documents loaded: {len(docs)}")
    
    if not docs:
        print("❌ No documents found or loaded")
        raise ValueError("No .txt or .pdf files found in the ingest path.")
    
    return docs

def build_vectorstore(docs, persist_dir: str = PERSIST_DIR) -> Chroma:
    print(f"[DEBUG] Building vectorstore with {len(docs)} documents")
    print(f"[DEBUG] Persist directory: {persist_dir}")
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        print(f"[DEBUG] Created {len(chunks)} chunks")
        
        for i, d in enumerate(chunks):
            meta = d.metadata or {}
            meta.setdefault("source", os.path.basename(meta.get("source", meta.get("file_path", "unknown"))))
            meta.setdefault("page", meta.get("page", meta.get("page_number", None)))
            meta["chunk_id"] = f"{meta.get('source','src')}#c{i}"
            d.metadata = meta
        
        print("[DEBUG] Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)
        print("[DEBUG] Creating Chroma vectorstore...")
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
        print("✓ Vectorstore created successfully")
        return vs
        
    except Exception as e:
        print(f"❌ Error building vectorstore: {e}")
        raise

def load_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    print(f"[DEBUG] Loading existing vectorstore from: {persist_dir}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBED_MODEL)
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("✓ Vectorstore loaded successfully")
        return vs
    except Exception as e:
        print(f"❌ Error loading vectorstore: {e}")
        raise

def format_evidence(docs: List, limit_preview: int = 300) -> str:
    lines = []
    for idx, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        preview = d.page_content.strip().replace("\n", " ")
        if len(preview) > limit_preview:
            preview = preview[:limit_preview] + "…"
        tag = f"[{idx}] {src}"
        if page is not None:
            tag += f", p.{page}"
        lines.append(f"{tag}: {preview}")
    return "\n".join(lines)

def extract_citation_indices(text: str, max_idx: int) -> List[int]:
    found = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= max_idx})
    return found

def create_llm() -> ChatGroq:
    print("[DEBUG] Creating LLM instance...")
    try:
        llm = ChatGroq(model=DEFAULT_LLM_MODEL, temperature=0.35, groq_api_key=GROQ_API_KEY)
        print("✓ LLM created successfully")
        return llm
    except Exception as e:
        print(f"❌ Error creating LLM: {e}")
        raise

def ai_respond(llm: ChatGroq, topic: str, opponent_text: str, evidence_docs: List) -> DebateTurn:
    print("[DEBUG] Generating AI response...")
    try:
        evidence_block = format_evidence(evidence_docs)
        prompt = (
            f"{DEBATE_AGENT_INSTRUCTIONS}\n\n"
            f"TOPIC: {topic}\n\n"
            f"OPPONENT: {opponent_text}\n\n"
            f"EVIDENCE:\n{evidence_block}\n\n"
            "Write a response that: (1) briefly rebuts the opponent using the evidence above, "
            "(2) provides the AI's own claim with 2-4 numbered supporting points. Cite evidence with [n]. "
            "Return plain text."
        )
        
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp)).strip()
        cited_idxs = extract_citation_indices(text, max_idx=len(evidence_docs))
        citations = []
        for i in cited_idxs:
            d = evidence_docs[i-1]
            citations.append({
                "n": i, 
                "source": d.metadata.get("source", "unknown"), 
                "page": d.metadata.get("page"), 
                "chunk_id": d.metadata.get("chunk_id")
            })
        
        print("✓ AI response generated successfully")
        return DebateTurn(actor="AI", text=text, citations=citations)
        
    except Exception as e:
        print(f"❌ Error generating AI response: {e}")
        raise

def interactive_debate(vs: Chroma, topic: str, rounds: int, save_path: Optional[str] = "debate_transcript.json"):
    print(f"[DEBUG] Starting interactive debate...")
    print(f"[DEBUG] Topic: {topic}")
    print(f"[DEBUG] Rounds: {rounds}")
    
    try:
        retriever = vs.as_retriever(search_kwargs={"k": 6})
        llm = create_llm()
        transcript: List[DebateTurn] = []

        print(f"\n=== Debate on: {topic} ===\n")
        
        for r in range(1, rounds + 1):
            print(f"--- Round {r} ---")
            human_arg = input("Your argument (single paragraph, press Enter): ").strip()
            
            if not human_arg:
                print("Empty argument — ending debate.")
                break
                
            transcript.append(DebateTurn(actor="HUMAN", text=human_arg, citations=[]))
            print(f"[DEBUG] Retrieving evidence for: {human_arg[:50]}...")
            
            evidence_docs = retriever.get_relevant_documents(human_arg)[:6]
            print(f"[DEBUG] Found {len(evidence_docs)} relevant documents")
            
            ai_turn = ai_respond(llm, topic, human_arg, evidence_docs)
            
            print("\nAI RESPONSE:\n")
            print(ai_turn.text)
            
            if ai_turn.citations:
                print("\nCitations:")
                for c in ai_turn.citations:
                    pg = f", p.{c['page']}" if c.get("page") else ""
                    print(f"[{c['n']}] {c['source']}{pg}")
            
            print("\n")
            transcript.append(ai_turn)

        dt = DebateTranscript(topic=topic, turns=transcript)
        
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "topic": dt.topic, 
                    "turns": [asdict(t) for t in dt.turns]
                }, f, ensure_ascii=False, indent=2)
            print(f"✓ Transcript saved to {save_path}")
            
    except Exception as e:
        print(f"❌ Error in interactive debate: {e}")
        raise

def main():
    print("[DEBUG] Starting main function...")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", type=str, default=None, help="Path to corpus to ingest (folder with txt/pdf).")
    ap.add_argument("--topic", type=str, required=True, help="Debate topic.")
    ap.add_argument("--rounds", type=int, default=3, help="Number of human↔AI rounds.")
    ap.add_argument("--out", type=str, default="debate_transcript.json", help="Transcript output file.")
    args = ap.parse_args()
    
    print(f"[DEBUG] Arguments parsed:")
    print(f"  - ingest: {args.ingest}")
    print(f"  - topic: {args.topic}")
    print(f"  - rounds: {args.rounds}")
    print(f"  - out: {args.out}")

    # Check API key
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY not set. Please set it as an environment variable.")
        print("   Example: export GROQ_API_KEY='your_actual_key'")
        sys.exit(1)
    else:
        print("✓ API key found")

    try:
        if args.ingest:
            docs = load_documents(args.ingest)
            vs = build_vectorstore(docs, persist_dir=PERSIST_DIR)
            print(f"✓ Built vector store at {PERSIST_DIR}")
        else:
            vs = load_vectorstore(PERSIST_DIR)
            
        interactive_debate(vs, args.topic, args.rounds, save_path=args.out)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
