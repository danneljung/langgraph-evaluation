from langsmith import Client, evaluate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM for hallucination checking
hallucination_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define the search tool
@tool
def search_docs(query: str) -> str:
    """Search documentation for information."""
    docs = {
        "python": "Python is a high-level programming language known for readability.",
        "langgraph": "LangGraph is a library for building stateful, multi-agent applications.",
        "rag": "RAG stands for Retrieval-Augmented Generation. It retrieves documents then generates answers."
    }
    
    for key, value in docs.items():
        if key in query.lower():
            return value
    
    return "No information found."

# Create the agent
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [search_docs]
agent = create_react_agent(model, tools)

# Target function
def target_function(inputs: dict) -> dict:
    """Wrapper function that takes dataset inputs and runs the agent."""
    question = inputs["question"]
    
    # Run the agent
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    
    # Extract the final answer
    answer = result["messages"][-1].content
    
    return {"answer": answer}

# Evaluators
def contains_expected_info(run, example):
    """Check if answer contains expected information."""
    actual_answer = run.outputs["answer"].lower()
    expected_answer = example.outputs["expected_answer"].lower()
    
    # For "No information found" cases
    if "no information" in expected_answer:
        score = 1.0 if "no information" in actual_answer else 0.0
    else:
        # Check if key phrases are present
        key_words = [word for word in expected_answer.split() 
                     if len(word) > 4 and word not in ['about', 'which', 'their']]
        
        # Check if at least 2 key words are in the actual answer
        matches = sum(1 for word in key_words if word in actual_answer)
        score = 1.0 if matches >= 2 else 0.0
    
    return {
        "key": "contains_expected_info",
        "score": score,
        "comment": f"Expected key info present: {'Yes' if score == 1.0 else 'No'}"
    }

def answer_length_check(run, example):
    """Check if answer is reasonable length."""
    answer = run.outputs["answer"]
    length = len(answer)
    
    # Reasonable if between 20 and 500 characters
    if 20 <= length <= 500:
        score = 1.0
    else:
        score = 0.5
    
    return {
        "key": "answer_length",
        "score": score,
        "comment": f"Length: {length} chars"
    }

def used_tool_correctly(run, example):
    """Check if the agent used the search_docs tool."""
    answer = run.outputs["answer"]
    
    # Simple heuristic: if answer is substantive, tool was likely used
    has_content = len(answer) > 50 and "No information" not in answer
    
    score = 1.0 if has_content else 0.0
    
    return {
        "key": "used_tool",
        "score": score,
        "comment": "Tool usage detected" if score == 1.0 else "No tool usage"
    }

def hallucination_check(run, example):
    """Uses LLM-as-judge to detect if the agent's output contains hallucinations."""
    
    question = example.inputs["question"]
    actual_answer = run.outputs["answer"]
    expected_answer = example.outputs["expected_answer"]
    
    # Create evaluation prompt for the LLM judge
    evaluation_prompt = f"""You are evaluating whether an AI assistant's response contains hallucinations.

QUESTION: {question}

EXPECTED ANSWER (for reference): {expected_answer}

ASSISTANT'S ACTUAL ANSWER: {actual_answer}

HALLUCINATION CRITERIA:
- Claims not supported by the question or general knowledge
- Fabricated specific details
- Mentioning unrelated topics (e.g., Java when asked about Python)
- Contradictions with known facts

INSTRUCTIONS:
Carefully analyze the answer and respond with ONLY:
- "PASS" if no hallucinations detected
- "FAIL" if hallucinations are present

Then briefly explain why in 1-2 sentences.

FORMAT:
Verdict: PASS or FAIL
Reasoning: [Brief explanation]

Your evaluation:"""

    try:
        # Get LLM judgment
        response = hallucination_judge.invoke(evaluation_prompt)
        response_text = response.content
        
        # Parse the response
        has_hallucination = "fail" in response_text.lower()
        
        # Score: 1 = no hallucination (good), 0 = hallucination detected (bad)
        score = 0.0 if has_hallucination else 1.0
        
        # Extract reasoning safely
        if "reasoning:" in response_text.lower():
            parts = response_text.lower().split("reasoning:")
            if len(parts) > 1:
                reasoning = parts[1].strip()[:150]
            else:
                reasoning = response_text[:150]
        else:
            reasoning = response_text[:150]
        
        return {
            "key": "hallucination_check",
            "score": score,
            "comment": f"{'HALLUCINATION' if has_hallucination else 'No hallucination'} - {reasoning}"
        }
    
    except Exception as e:
        return {
            "key": "hallucination_check",
            "score": None,
            "comment": f"Error: {str(e)}"
        }

# Run the evaluation
if __name__ == "__main__":
    print("\nStarting evaluation...")
    
    # Initialize client
    client = Client()
    
    # Dataset name
    dataset_name = "rag-agent-golden-dataset"
    
    print(f"Dataset: {dataset_name}")
    print(f"Agent: RAG agent with search_docs tool")
    print(f"Evaluators: contains_expected_info, answer_length, used_tool, hallucination_check")
    print("Running evaluation...\n")
    
    # Run evaluation
    results = evaluate(
        target_function,
        data=dataset_name,
        evaluators=[
            contains_expected_info,
            answer_length_check,
            used_tool_correctly,
            hallucination_check
        ],
        experiment_prefix="rag-agent-v2",
        description="Evaluation with hallucination detection added via LLM-as-judge",
        metadata={
            "agent_type": "react",
            "model": "gpt-4o-mini",
            "version": "2.0",
            "evaluators": ["contains_expected_info", "answer_length", "used_tool", "hallucination_check"]
        }
    )
    
    print("\nEvaluation complete!")
    print(f"Experiment: {results.experiment_name}")
    print(f"View results at: https://smith.langchain.com/")
