"""
Test script for KnowledgeAugmentedPromptAgent

This script demonstrates how to initialize and use the KnowledgeAugmentedPromptAgent
from the base_agents module.
"""
import os
import sys
from dotenv import load_dotenv


from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load API key from environment variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Define a persona and knowledge for the agent
persona = "a cloud computing specialist"
knowledge = """
Amazon Web Services (AWS) offers the following key services:
1. EC2 (Elastic Compute Cloud) - Virtual servers in the cloud
2. S3 (Simple Storage Service) - Object storage service
3. RDS (Relational Database Service) - Managed database service
4. Lambda - Serverless computing platform
5. CloudFront - Content delivery network service
6. IAM (Identity and Access Management) - Manages access to AWS services

Microsoft Azure offers the following key services:
1. Azure Virtual Machines - Compute services
2. Azure Blob Storage - Object storage service
3. Azure SQL Database - Managed database service
4. Azure Functions - Serverless computing service
5. Azure Content Delivery Network - Content delivery network service
6. Azure Active Directory - Identity and access management service
"""

# Initialize the KnowledgeAugmentedPromptAgent
agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# Sample prompt
prompt = "What are the main differences between AWS S3 and Azure Blob Storage?"

# Get response from agent
print("Sending prompt to KnowledgeAugmentedPromptAgent...")
print(f"Persona: '{persona}'")
print("Knowledge provided about AWS and Azure cloud services")
print(f"Prompt: '{prompt}'")
print("\nKnowledgeAugmentedPromptAgent response:")
response = agent.respond(prompt)
print(response)

print("\nNote: KnowledgeAugmentedPromptAgent uses a system prompt to set its persona")
print("and provides specific knowledge that the agent must use to answer questions.")
print("The agent is instructed to use only the provided knowledge and forget any prior knowledge.")
