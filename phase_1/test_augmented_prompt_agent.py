"""
Test script for AugmentedPromptAgent

This script demonstrates how to initialize and use the AugmentedPromptAgent
from the base_agents module.
"""
import os
import sys
from dotenv import load_dotenv

from workflow_agents.base_agents import AugmentedPromptAgent

# Load API key from environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Define a persona for the agent
persona = "a helpful career counselor who provides guidance on professional development"

# Initialize the AugmentedPromptAgent
agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)

# Sample prompt
prompt = "What skills should I develop to advance in a data science career?"

# Get response from agent
print("Sending prompt to AugmentedPromptAgent...")
print(f"Persona: '{persona}'")
print(f"Prompt: '{prompt}'")
print("\nAugmentedPromptAgent response:")
response = agent.respond(prompt)
print(response)

print("\nNote: AugmentedPromptAgent uses a system prompt to set its persona")
print("and instructs the LLM to forget any previous conversational context.")
print("The response is influenced by the persona provided during initialization.")
