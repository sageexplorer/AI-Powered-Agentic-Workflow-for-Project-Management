"""
Test script for DirectPromptAgent

This script demonstrates how to initialize and use the DirectPromptAgent
from the base_agents module.
"""
import os
import sys
from dotenv import load_dotenv

from workflow_agents.base_agents import DirectPromptAgent

# Load API key from environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Initialize the DirectPromptAgent
agent = DirectPromptAgent(openai_api_key=openai_api_key)

# Sample prompt
prompt = "Explain what a DirectPromptAgent is in simple terms."

# Get response from agent
print("Sending prompt to DirectPromptAgent...")
print(f"Prompt: '{prompt}'")
print("\nDirectPromptAgent response:")
response = agent.respond(prompt)
print(response)
print("\nNote: DirectPromptAgent does not use a system prompt or persona.")
print("It passes the user prompt directly to the LLM without any additional context or instructions.")
