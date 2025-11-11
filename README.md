# AgentVille: AI-Powered Agentic Workflow for Project Management

This project demonstrates the implementation of AI-powered agentic workflows for project management tasks. It showcases various agent architectures and workflow patterns that can be used to automate and enhance project management processes.

## Security Notice

⚠️ **Important:** This project uses API keys for OpenAI services. The `.env` file contains sensitive API keys that should not be committed to version control. Before committing any code:

1. Remove any API keys from the `.env` file
2. Add `.env` to your `.gitignore` file
3. Use a template `.env.example` file to show the required environment variables without actual values

## Project Structure

The project is organized into two main phases:

### Phase 1: Agent Implementation and Testing

Contains the base agent implementations and individual test files for each agent type:

- `workflow_agents/base_agents.py` - Core agent implementations
- Test files for various agent implementations:
  - `test_direct_prompt_agent.py`
  - `test_augmented_prompt_agent.py`
  - `test_knowledge_augmented_prompt_agent.py`
  - `test_rag_agent.py`
  - `test_evaluation_agent.py`
  - `test_routing_agent.py`
  - `test_action_planning_agent.py`
- `test_results_output.txt` - Results of agent tests

### Phase 2: Agentic Workflow Implementation

Demonstrates a complete agentic workflow for creating project plans:

- `agentic_workflow.py` - Main workflow implementation
- `Product-Spec-Email-Router.txt` - Sample product specification
- `workflow_agents/base_agents.py` - Reused agent implementations
- `result_output.txt` - Output of the workflow execution

### Root Directory

Contains various tools and utilities:

- `run_rag_test.py` - Script to test RAG functionality
- `troubleshoot_rag_test.py` - Script to debug RAG issues
- Various workflow pattern implementations:
  - `evaluator_optimizer_workflow_pattern.py`
  - `prompt_chaining.py`
  - `routing.py`
  - `parallel.py`
  - `parallel_enterprise_contract.py`

## Agent Types

The project implements several types of agents:

1. **DirectPromptAgent**: Basic agent that forwards prompts directly to the language model
2. **AugmentedPromptAgent**: Adds a persona to guide language model responses
3. **KnowledgeAugmentedPromptAgent**: Uses predefined knowledge to answer questions
4. **RAGKnowledgePromptAgent**: Uses Retrieval-Augmented Generation to find information in a corpus
5. **EvaluationAgent**: Evaluates responses against criteria and provides feedback
6. **RoutingAgent**: Routes prompts to the most appropriate specialized agent
7. **ActionPlanningAgent**: Breaks down tasks into actionable steps

## Workflow Patterns

The project demonstrates several workflow patterns:

1. **Evaluation-Optimization**: Evaluates responses and iteratively improves them
2. **Routing**: Directs prompts to specialized agents based on content
3. **Action Planning**: Breaks down complex tasks into sequential steps
4. **Prompt Chaining**: Passes results from one agent to another
5. **Parallel Processing**: Executes multiple agent tasks concurrently

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sageexplorer/AI-Powered-Agentic-Workflow-for-Project-Management.git
cd AgentVille
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Testing Individual Agents

Run tests for specific agents:

```bash
# Test RAG agent
python run_rag_test.py

# Test other agents
python phase_1/test_direct_prompt_agent.py
python phase_1/test_augmented_prompt_agent.py
# etc.
```

### Running the Complete Workflow

Execute the agentic workflow for project planning:

```bash
cd phase_2
python agentic_workflow.py
```

The workflow will:
1. Extract actionable steps from a workflow prompt
2. Route each step to appropriate specialized agents
3. Validate and refine the responses
4. Produce a complete project plan with user stories, features, and tasks

## Example: Email Router Project Plan

The Phase 2 workflow demonstrates creating a project plan for an Email Router system with:
- User stories following the format "As a [user], I want [action] so that [benefit]"
- Product features with names, descriptions, functionality, and user benefits
- Engineering tasks with IDs, titles, descriptions, and acceptance criteria

## Contributing

Contributions are welcome! Please ensure:
1. No API keys or sensitive information in commits
2. Code follows the existing structure and style
3. New features include appropriate tests

## License


