from datasets import load_dataset
from typing import List, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_squad_v2(split='validation'):
    """Load SQuAD v2 dataset and convert to list of examples."""
    dataset = load_dataset("squad_v2", split=split)
    examples = []
    for item in dataset:
        examples.append({
            'id': item['id'],
            'title': item['title'],
            'context': item['context'],
            'question': item['question'],
            'answers': item['answers'],
            'is_impossible': item['answers']['text'] == []
        })
    return examples

def prepare_contexts_for_rag(examples: List[Dict], chunk_size=1000, chunk_overlap=200):
    """Convert SQuAD contexts into document chunks for vector store."""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    for ex in examples:
        doc = Document(
            page_content=ex['context'],
            metadata={
                'id': ex['id'],
                'title': ex['title'],
                'question': ex['question'],
                'is_impossible': ex['is_impossible']
            }
        )
        chunks = text_splitter.split_documents([doc])
        documents.extend(chunks)
    
    return documents

