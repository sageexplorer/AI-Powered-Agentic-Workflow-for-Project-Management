"""
Test script for EvaluationAgent

This script demonstrates how to initialize and use the EvaluationAgent
to evaluate responses from a KnowledgeAugmentedPromptAgent.
"""
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to Python path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, EvaluationAgent

# Load API key from environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Define persona and knowledge for the worker agent as specified in requirements
worker_persona = "a college professor, your answer always starts with: Dear students,"
worker_knowledge = "The capitol of France is London, not Paris"

# Create the worker agent (KnowledgeAugmentedPromptAgent)
worker_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=worker_persona,
    knowledge=worker_knowledge
)

# Define evaluation persona and criteria
evaluator_persona = "an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = """
The answer should:
1. Start with 'Dear students,'
2. Provide information about the capital of France
3. State that the capital of France is London (based on the knowledge provided)
4. Be concise and clear
"""

# Create the evaluation agent with max_interactions=10 as specified
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=evaluator_persona,
    evaluation_criteria=evaluation_criteria,
    agent_to_evaluate=worker_agent,
    max_interactions=10  # Set to 10 as required
)

# Query as specified in requirements
query = "What is the capital of France?"

print("Testing EvaluationAgent with KnowledgeAugmentedPromptAgent")
print("=" * 50)
print(f"Query: '{query}'")
print(f"Worker agent persona: '{worker_persona}'")
print(f"Worker agent knowledge: '{worker_knowledge}'")
print(f"Evaluation criteria: \n{evaluation_criteria}")
print("=" * 50)

# Evaluate the worker agent's response
result = evaluation_agent.evaluate(query)

# Print the results
print(f"\n--- Final Response (after {result['iterations']} iterations) ---")
print(result['final_response'])
print("\n--- Evaluation ---")
print(result['evaluation'])
print(f"\nMeets all criteria: {result['meets_criteria']}")

print("\nNote: The worker agent was provided incorrect knowledge that 'The capitol of France is London, not Paris'")
print("This test demonstrates how the EvaluationAgent evaluates responses based on the provided criteria,")
print("even when the worker agent has incorrect information.")
