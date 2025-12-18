"""
main.py - SQuAD v2 Dataset Evaluation
Complete evaluation pipeline using dataset_loader.py and metrics.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
from tqdm import tqdm
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

# Import existing modules
from dataset_loader import load_squad_v2, prepare_contexts_for_rag
from metrics import evaluate_batch, evaluate_unanswerable

# CONFIGURATION
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_RETRIEVED = 3
MAX_SAMPLES = 50  # Set to None to use all samples

# RAG SETUP
def format_docs(docs):
    """Format retrieved documents for context"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Source {i}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def setup_rag_system(contexts):
    """Setup complete RAG system with retriever and LLM"""

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=contexts,
        embedding=embedding_model
    )
    
    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K_RETRIEVED}
    )
    
    # LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0
    )
    
    # Prompt 
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a question-answering assistant. "
            "Answer the question using ONLY the provided context. "
            "If the answer cannot be found in the context, respond with 'I don't know' or 'The answer is not available in the provided context.'"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
    ])
    
    # RAG Chain
    rag_chain = (
        {
            "docs": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "context": format_docs(x["docs"]),
        })
        | RunnableLambda(lambda x: {
            "answer": (
                prompt
                | llm
                | StrOutputParser()
            ).invoke({
                "question": x["question"],
                "context": x["context"],
            })
        })
    )
    
    print("RAG system ready!")
    return rag_chain

# EVALUATION
def evaluate_on_squad(rag_chain, examples, max_samples=MAX_SAMPLES):
    """Evaluate RAG system """
    if max_samples:
        examples = examples[:max_samples]
        print(f"Using {max_samples} samples for evaluation")
    
    predictions = []
    ground_truths_list = []
    is_impossible_list = []
    
    print(f"\nEvaluating on {len(examples)} examples...")
    
    for example in tqdm(examples, desc="Processing"):
        try:
            # Get prediction from RAG system
            result = rag_chain.invoke({"question": example['question']})
            prediction = result["answer"]
            predictions.append(prediction)
            
            # Get ground truth answers
            ground_truths = example['answers']['text']
            ground_truths_list.append(ground_truths)
            
            # Track if unanswerable
            is_impossible_list.append(example['is_impossible'])
        except Exception as e:
            print(f"Error processing example {example['id']}: {e}")
            predictions.append("")
            ground_truths_list.append(example['answers']['text'])
            is_impossible_list.append(example['is_impossible'])
    
    # Evaluate using metrics.py
    metrics = evaluate_batch(predictions, ground_truths_list)
    unanswerable_metrics = evaluate_unanswerable(predictions, is_impossible_list)
    
    # Combine results
    results = {
        **metrics,
        **unanswerable_metrics,
        'total_samples': len(examples),
        'answerable_samples': sum(1 - x for x in is_impossible_list),
        'unanswerable_samples': sum(is_impossible_list)
    }
    
    return results, predictions


# MAIN EXECUTION
def main():
    """Main evaluation pipeline"""
    print("SQuAD v2 RAG Evaluation Pipeline")
    
    # Step 1: Load SQuAD v2 dataset (using your dataset_loader.py)
    print("\nStep 1: Loading dataset...")
    examples = load_squad_v2(split='validation')
    print(f"Loaded {len(examples)} examples")
    
    # Step 2: Prepare contexts for RAG (using your dataset_loader.py)
    print("\nStep 2: Preparing contexts...")
    contexts = prepare_contexts_for_rag(
        examples,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print(f"Created {len(contexts)} document chunks")
    
    # Step 3: Setup RAG system
    print("\nStep 3: Setting up RAG system...")
    rag_chain = setup_rag_system(contexts)
    
    # Step 4: Run evaluation
    print("\nStep 4: Running evaluation...")
    results, predictions = evaluate_on_squad(rag_chain, examples, MAX_SAMPLES)
    
    # Step 5: Display results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"F1 Score:              {results['f1']:.4f}")
    print(f"Exact Match (EM):      {results['em']:.4f}")
    print(f"Unanswerable Detection: {results['unanswerable_detection_accuracy']:.4f}")
    print(f"\nDataset Statistics:")
    print(f"  Total Samples:       {results['total_samples']}")
    print(f"  Answerable:          {results['answerable_samples']}")
    print(f"  Unanswerable:        {results['unanswerable_samples']}")
    print("="*60)
    
    # Step 6: Save results
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': results,
            'sample_predictions': predictions[:20],  # Save first 20 for inspection
            'config': {
                'embedding_model': EMBEDDING_MODEL,
                'llm_model': LLM_MODEL,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP,
                'k_retrieved': K_RETRIEVED,
                'max_samples': MAX_SAMPLES
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()