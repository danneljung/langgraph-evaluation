from langsmith import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

# Define test cases (our "golden dataset")
test_cases = [
    {
        "question": "What is Python?",
        "expected_answer": "Python is a high-level programming language known for readability.",
        "category": "basic"
    },
    {
        "question": "What is LangGraph?",
        "expected_answer": "LangGraph is a library for building stateful, multi-agent applications.",
        "category": "basic"
    },
    {
        "question": "What is RAG?",
        "expected_answer": "RAG stands for Retrieval-Augmented Generation. It retrieves documents then generates answers.",
        "category": "basic"
    },
    {
        "question": "What is Java?",
        "expected_answer": "No information found.",
        "category": "edge_case"
    },
    {
        "question": "Tell me about LangGraph and its purpose",
        "expected_answer": "LangGraph is a library for building stateful, multi-agent applications.",
        "category": "variation"
    },
]

# Dataset name
dataset_name = "rag-agent-golden-dataset"

# Create or update dataset
try:
    # Try to read existing dataset
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"Dataset '{dataset_name}' already exists")
    print(f"Dataset ID: {dataset.id}")
except:
    # Create new dataset
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Golden dataset for RAG agent evaluation - contains verified Q&A pairs"
    )
    print(f"Created new dataset: {dataset_name}")
    print(f"Dataset ID: {dataset.id}")
    
    # Add examples to dataset
    for i, case in enumerate(test_cases, 1):
        client.create_example(
            inputs={"question": case["question"]},
            outputs={"expected_answer": case["expected_answer"]},
            metadata={"category": case["category"]},
            dataset_id=dataset.id
        )
        print(f"Added example {i}/{len(test_cases)}: {case['question'][:50]}...")
    
    print(f"\nAdded {len(test_cases)} examples to dataset")

print(f"\nView dataset at: https://smith.langchain.com/")
print(f"Dataset creation complete!")
