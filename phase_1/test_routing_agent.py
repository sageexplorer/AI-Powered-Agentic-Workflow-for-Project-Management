"""
Test script for RoutingAgent

This script demonstrates how to initialize and use the RoutingAgent
from the base_agents module to route prompts to the most appropriate agent.
"""
import os
import sys
from dotenv import load_dotenv

from workflow_agents.base_agents import RoutingAgent, KnowledgeAugmentedPromptAgent

# Load API key from environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Create worker agents for different domains

# Product Manager Agent
product_manager_knowledge = """
A Product Manager is responsible for:
1. Defining the product vision and roadmap
2. Gathering and prioritizing user requirements
3. Working with development teams to deliver the product
4. Monitoring product performance and user feedback
5. Making data-driven decisions to improve the product
"""
product_manager = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="a product manager with 10 years of experience in software product development",
    knowledge=product_manager_knowledge
)

# Technical Support Agent
tech_support_knowledge = """
Technical support best practices include:
1. Asking clear diagnostic questions
2. Following a structured troubleshooting process
3. Documenting all steps taken
4. Explaining technical issues in simple terms
5. Following up to ensure issues are resolved
"""
tech_support = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="a technical support specialist with expertise in software troubleshooting",
    knowledge=tech_support_knowledge
)

# Marketing Agent
marketing_knowledge = """
Effective marketing strategies include:
1. Understanding target audience demographics and behaviors
2. Creating compelling value propositions
3. Selecting appropriate marketing channels
4. Measuring campaign effectiveness
5. Optimizing based on performance data
"""
marketing = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="a marketing professional with expertise in digital marketing campaigns",
    knowledge=marketing_knowledge
)

# Define agent configurations for the routing agent
agents = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product features, user stories, and roadmaps. Can help with product strategy, requirement gathering, and feature prioritization.",
        "func": lambda prompt: product_manager.respond(prompt)
    },
    {
        "name": "Technical Support",
        "description": "Helps troubleshoot technical issues, provides step-by-step solutions to problems, and assists with error resolution and system diagnostics.",
        "func": lambda prompt: tech_support.respond(prompt)
    },
    {
        "name": "Marketing Specialist",
        "description": "Expert in marketing campaigns, customer acquisition, branding, and market analysis. Can help with marketing strategy and promotional content.",
        "func": lambda prompt: marketing.respond(prompt)
    }
]

# Create the routing agent
routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=agents
)

# Sample prompts for testing
prompts = [
    "We need to define the key features for our new mobile app. What should be our priorities?",
    "Our website keeps crashing when users try to upload files. How can we diagnose this issue?",
    "I need ideas for promoting our new product launch on social media."
]

# Test the routing agent
print("Testing RoutingAgent with different prompts...\n")

for i, prompt in enumerate(prompts, 1):
    print(f"--- Test {i}: ---")
    print(f"Prompt: '{prompt}'")
    
    print("\nRouting and generating response...")
    response = routing_agent.route(prompt)
    
    print("\nResponse:")
    print(response)
    print("\n" + "-"*50 + "\n")

print("Note: RoutingAgent uses text-embedding-3-large to compute embeddings")
print("for both the prompt and agent descriptions, then routes the prompt")
print("to the most appropriate agent based on cosine similarity.")
