"""
Test script for ActionPlanningAgent

This script demonstrates how to initialize and use the ActionPlanningAgent
from the base_agents module.
"""
import os
import sys
from dotenv import load_dotenv


from workflow_agents.base_agents import ActionPlanningAgent

# Load API key from environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Optional knowledge to guide the planning process
knowledge = """
Software development best practices include:
- Breaking down tasks into small, manageable pieces
- Writing clean, readable code with proper documentation
- Implementing automated tests
- Using version control
- Conducting code reviews
- Following continuous integration/continuous deployment principles
"""

# Initialize the ActionPlanningAgent
agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge
)

# Sample prompt
prompt = "Create a simple web application that allows users to track their daily exercise routines"

# Get actionable steps from agent
print("Sending prompt to ActionPlanningAgent...")
print(f"Prompt: '{prompt}'")
print("\nActionable steps extracted by ActionPlanningAgent:")
steps = agent.extract_steps_from_prompt(prompt)

# Print the steps
for i, step in enumerate(steps, 1):
    print(f"{i}. {step}")

print("\nNote: ActionPlanningAgent extracts actionable steps from a user prompt")
print("and returns them as a clean, structured list.")
print("The agent is designed to break down complex tasks into logical, sequential steps.")
