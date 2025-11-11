
import os
from dotenv import load_dotenv
from workflow_agents import base_agents


openai_api_key = os.getenv("OPENAI_API_KEY")


with open("Product-Spec-Email-Router.txt", "r") as file:
    product_spec = file.read()

# Instantiate all the agents
# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)


action_planning_agent = base_agents.ActionPlanningAgent(openai_api_key, knowledge_action_planning)


# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    + product_spec
)

# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = base_agents.KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# Define evaluation criteria for user stories
evaluation_criteria_product_manager = (
    "The answer should be user stories that follow the exact structure: " 
    "'As a [type of user], I want [an action or feature] so that [benefit/value].' " 
    "Each user story should clearly identify: " 
    "1. The specific type of user (persona) " 
    "2. The action or feature they want " 
    "3. The benefit or value they receive " 
    "Stories should be concise, focused on a single functionality, " 
    "and align with the product spec requirements."
)

# Instantiate the Product Manager evaluation agent
product_manager_evaluation_agent = base_agents.EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,
)



# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."

# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = base_agents.KnowledgeAugmentedPromptAgent(
    openai_api_key, 
    persona_program_manager, 
    knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_program_manager = ("The answer should be product features that follow the following structure: " \
                     "Feature Name: A clear, concise title that identifies the capability\n" \
                     "Description: A brief explanation of what the feature does and its purpose\n" \
                     "Key Functionality: The specific capabilities or actions the feature provides\n" \
                     "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = base_agents.EvaluationAgent(openai_api_key, persona_program_manager_eval, evaluation_criteria_program_manager, program_manager_knowledge_agent)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."

# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'

development_engineer_knowledge_agent = base_agents.KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_development_engineer = (
                     "The answer should be tasks following this exact structure: " \
                     "Task ID: A unique identifier for tracking purposes\n" \
                     "Task Title: Brief description of the specific development work\n" \
                     "Related User Story: Reference to the parent user story\n" \
                     "Description: Detailed explanation of the technical work required\n" \
                     "Acceptance Criteria: Specific requirements that must be met for completion\n" \
                     "Estimated Effort: Time or complexity estimation\n" \
                     "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = base_agents.EvaluationAgent(openai_api_key, persona_dev_engineer_eval, evaluation_criteria_development_engineer, development_engineer_knowledge_agent)


def product_manager_support_function(query):
    """
    Support function for the Product Manager role.
    
    Args:
        query (str): The input query (a step from the action plan).
        
    Returns:
        str: The validated response from the Product Manager.
    """
    # 1. Get a response from the Knowledge Augmented Prompt Agent
    response = product_manager_knowledge_agent.respond(query)
    
    # 2. Have the response evaluated by the Evaluation Agent
    # Pass the query as context and the response to evaluate
    evaluation_result = product_manager_evaluation_agent.evaluate(f"For the query: {query}\n\nResponse to evaluate: {response}")
    
    # 3. Return the final validated response
    return evaluation_result["final_response"]

def program_manager_support_function(query):
    """
    Support function for the Program Manager role.
    
    Args:
        query (str): The input query (a step from the action plan).
        
    Returns:
        str: The validated response from the Program Manager.
    """
    # 1. Get a response from the Knowledge Augmented Prompt Agent
    response = program_manager_knowledge_agent.respond(query)
    
    # 2. Have the response evaluated by the Evaluation Agent
    # Pass the query as context and the response to evaluate
    evaluation_result = program_manager_evaluation_agent.evaluate(f"For the query: {query}\n\nResponse to evaluate: {response}")
    
    # 3. Return the final validated response
    return evaluation_result["final_response"]

def development_engineer_support_function(query):
    """
    Support function for the Development Engineer role.
    
    Args:
        query (str): The input query (a step from the action plan).
        
    Returns:
        str: The validated response from the Development Engineer.
    """
    # 1. Get a response from the Knowledge Augmented Prompt Agent
    response = development_engineer_knowledge_agent.respond(query)
    
    # 2. Have the response evaluated by the Evaluation Agent
    # Pass the query as context and the response to evaluate
    evaluation_result = development_engineer_evaluation_agent.evaluate(f"For the query: {query}\n\nResponse to evaluate: {response}")
    
    # 3. Return the final validated response
    return evaluation_result["final_response"]


routes = [
    {
        'name': 'Product Manager',
        'description': 'Defines user stories based on product requirements. Use this for tasks related to creating user stories, understanding user needs, and defining product behavior from a user perspective.',
        'func': product_manager_support_function
    },
    {
        'name': 'Program Manager',
        'description': 'Organizes user stories into coherent product features. Use this for tasks related to feature definition, grouping functionality, and organizing product capabilities.',
        'func': program_manager_support_function
    },
    {
        'name': 'Development Engineer',
        'description': 'Defines technical tasks needed to implement user stories. Use this for tasks related to technical implementation, development work breakdown, and engineering specifications.',
        'func': development_engineer_support_function
    }
]

# Routing Agent
routing_agent = base_agents.RoutingAgent(openai_api_key, routes)

# Routes already assigned during initialization

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "Create a concise project plan for the Email Router system with exactly: 1) Three user stories following the structure 'As a [type of user], I want [an action or feature] so that [benefit/value]', 2) Three product features following the structure 'Feature Name: ...', 'Description: ...', 'Key Functionality: ...', 'User Benefit: ...', and 3) Three engineering tasks following the structure 'Task ID: ...', 'Task Title: ...', 'Related User Story: ...', 'Description: ...', 'Acceptance Criteria: ...', 'Estimated Effort: ...', 'Dependencies: ...'"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Action planning agent extracted these steps: {workflow_steps}")

# 2. Initialize an empty list to store 'completed_steps'.
completed_steps = []

# 3. Loop through the extracted workflow steps
print("\nExecuting workflow steps:")
for i, step in enumerate(workflow_steps):
    # a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
    print(f"\nStep {i+1}: {step}")
    step_result = routing_agent.route(step)
    
    # b. Append the result to 'completed_steps'.
    completed_steps.append(step_result)
    
    # c. Print information about the step being executed and its result.
    print(f"Result: {step_result[:100]}..." if len(step_result) > 100 else f"Result: {step_result}")

# 4. Combine all completed steps for a comprehensive plan
print("\n*** Workflow execution completed ***\n")
if completed_steps:
    # Get all the completed steps
    user_stories = []
    product_features = []
    engineering_tasks = []
    
    for step in completed_steps:
        # Look for user stories (in the format "As a [type of user]...")
        if "As a " in step and "I want" in step and "so that" in step:
            user_stories.append(step)
        # Look for product features (in the format "Feature Name:")
        elif "Feature Name:" in step or ("Description:" in step and "Key Functionality:" in step and "User Benefit:" in step):
            product_features.append(step)
        # Look for engineering tasks (in the format "Task ID:")
        elif "Task ID:" in step and "Task Title:" in step and "Description:" in step:
            engineering_tasks.append(step)
    
    print("Final Project Plan for the Email Router System:")
    print("\n==== USER STORIES ====\n")
    if user_stories:
        for i, story in enumerate(user_stories):
            print(f"User Story {i+1}:\n{story}\n")
    else:
        print("No user stories were generated.")
        
    print("\n==== PRODUCT FEATURES ====\n")
    if product_features:
        for i, feature in enumerate(product_features):
            print(f"Product Feature {i+1}:\n{feature}\n")
    else:
        print("No product features were generated.")
        
    print("\n==== ENGINEERING TASKS ====\n")
    if engineering_tasks:
        for i, task in enumerate(engineering_tasks):
            print(f"Engineering Task {i+1}:\n{task}\n")
    else:
        print("No engineering tasks were generated.")
else:
    print("No steps were completed in the workflow.")
