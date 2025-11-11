import numpy as np
import pandas as pd
import re
import csv
import uuid
import traceback
import logging
from datetime import datetime
from openai import OpenAI 



# DirectPromptAgent class definition
class DirectPromptAgent:
    
    def __init__(self, openai_api_key):
        # Initialize the agent
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content

        

# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.openai_api_key = openai_api_key
        self.persona = persona 

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.persona},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return  response.choices[0].message.content 



# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key)

        system_message = f"""
                        You are {self.persona} knowledge-based assistant. Forget all previous context.
                        Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}
                        Answer the prompt based on this knowledge, not your own.
                         """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        # Initialize logging
        import logging
        logging.basicConfig(
            filename="rag_debug.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("RAGKnowledgePromptAgent")
        self.logger.info(f"Agent initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        try:
            self.logger.info(f"Getting embedding for text of length {len(text)}")
            client = OpenAI(api_key=self.openai_api_key)
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                encoding_format="float"
            )
            self.logger.info("Successfully retrieved embedding")
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            raise

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        try:
            vec1, vec2 = np.array(vector_one), np.array(vector_two)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            self.logger.debug(f"Calculated similarity: {similarity}")
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        try:
            self.logger.info(f"Chunking text of length {len(text)}")
            separator = "\n"
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) <= self.chunk_size:
                self.logger.info("Text fits in a single chunk")
                return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

            chunks, start, chunk_id = [], 0, 0
            self.logger.info(f"Starting chunking with size={self.chunk_size}, overlap={self.chunk_overlap}")

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                if separator in text[start:end]:
                    end = start + text[start:end].rindex(separator) + len(separator)
                
                chunk_text = text[start:end]
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end
                })
                
                self.logger.debug(f"Created chunk {chunk_id}: length={end-start}, start={start}, end={end}")
                
                start = end - self.chunk_overlap
                chunk_id += 1

            self.logger.info(f"Created {len(chunks)} chunks")

            # Save chunks to CSV
            chunk_file = f"chunks-{self.unique_filename}"
            self.logger.info(f"Saving chunks to {chunk_file}")
            with open(chunk_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
                writer.writeheader()
                for chunk in chunks:
                    writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

            return chunks
        except Exception as e:
            self.logger.error(f"Error in chunk_text: {str(e)}")
            raise

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        try:
            chunk_file = f"chunks-{self.unique_filename}"
            embeddings_file = f"embeddings-{self.unique_filename}"
            
            self.logger.info(f"Loading chunks from {chunk_file}")
            df = pd.read_csv(chunk_file, encoding='utf-8')
            self.logger.info(f"Loaded {len(df)} chunks, calculating embeddings")
            
            # Check if there are too many chunks
            if len(df) > 20:
                self.logger.warning(f"Large number of chunks detected ({len(df)}), this may cause memory issues")
            
            # Calculate embeddings one by one with batching to avoid memory issues
            self.logger.info("Calculating embeddings with batching")
            batch_size = 5  # Process 5 chunks at a time
            all_embeddings = []
            
            for i in range(0, len(df), batch_size):
                batch_end = min(i + batch_size, len(df))
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
                
                batch_embeddings = []
                for j in range(i, batch_end):
                    try:
                        self.logger.info(f"Processing chunk {j+1}/{len(df)}")
                        text = df.loc[j, 'text']
                        embedding = self.get_embedding(text)
                        batch_embeddings.append(embedding)
                    except Exception as e:
                        self.logger.error(f"Error embedding chunk {j+1}: {str(e)}")
                        # Use a placeholder embedding if there's an error
                        batch_embeddings.append(np.zeros(1536))  # Assuming 1536-dimension embeddings
                
                all_embeddings.extend(batch_embeddings)
            
            # Add embeddings to dataframe
            df['embeddings'] = all_embeddings
            
            # Save to CSV
            self.logger.info(f"Saving embeddings to {embeddings_file}")
            df.to_csv(embeddings_file, encoding='utf-8', index=False)
            self.logger.info("Embeddings calculation complete")
            return df
        except Exception as e:
            self.logger.error(f"Error in calculate_embeddings: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Create a minimal valid dataframe if there's an error
            try:
                # Create a minimal dataframe with just one chunk and embedding
                self.logger.info("Creating fallback dataframe due to error")
                df_fallback = pd.DataFrame({
                    'text': ["Fallback text due to error"],
                    'chunk_size': [len("Fallback text due to error")],
                    'embeddings': [np.zeros(1536).tolist()]  # Placeholder embedding
                })
                df_fallback.to_csv(embeddings_file, encoding='utf-8', index=False)
                return df_fallback
            except Exception as nested_e:
                self.logger.error(f"Even fallback failed: {str(nested_e)}")
                raise

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        try:
            self.logger.info(f"Processing prompt: '{prompt}'")
            
            # Get embedding for the prompt
            self.logger.info("Getting prompt embedding")
            prompt_embedding = self.get_embedding(prompt)
            
            # Load embeddings file
            embeddings_file = f"embeddings-{self.unique_filename}"
            self.logger.info(f"Loading embeddings from {embeddings_file}")
            
            # Use memory-efficient loading with chunking
            try:
                # Try with optimized loading for large files
                self.logger.info("Using memory-efficient dataframe loading")
                # Read in chunks to avoid memory issues
                df_chunks = []
                for chunk in pd.read_csv(embeddings_file, encoding='utf-8', chunksize=10):
                    df_chunks.append(chunk)
                df = pd.concat(df_chunks, ignore_index=True)
            except Exception as chunk_e:
                self.logger.warning(f"Chunked loading failed: {str(chunk_e)}. Trying regular loading.")
                df = pd.read_csv(embeddings_file, encoding='utf-8')
            
            self.logger.info(f"Loaded embeddings dataframe with {len(df)} rows")
            
            # Process embeddings with error handling for each row
            self.logger.info("Converting stored embeddings to numpy arrays")
            embeddings = []
            for i, row in df.iterrows():
                try:
                    embedding_str = row['embeddings']
                    embedding = np.array(eval(embedding_str))
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.error(f"Error processing embedding at row {i}: {str(e)}")
                    # Use zero vector as fallback
                    embeddings.append(np.zeros(1536))
            
            df['processed_embeddings'] = embeddings
            
            # Calculate similarities
            self.logger.info("Calculating similarities between prompt and chunks")
            similarities = []
            for emb in df['processed_embeddings']:
                try:
                    similarity = self.calculate_similarity(prompt_embedding, emb)
                    similarities.append(similarity)
                except Exception as e:
                    self.logger.error(f"Error calculating similarity: {str(e)}")
                    similarities.append(0.0)  # Default to 0 similarity on error
            
            df['similarity'] = similarities
            
            # Find best chunk
            if len(df) == 0:
                self.logger.error("No valid chunks found")
                return "No valid knowledge chunks found to answer the question."
                
            best_index = df['similarity'].idxmax()
            best_score = df.loc[best_index, 'similarity']
            best_chunk = df.loc[best_index, 'text']
            self.logger.info(f"Best chunk found at index {best_index} with similarity {best_score}")
            
            # Generate response
            self.logger.info("Generating response with GPT")
            client = OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                    {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content
            self.logger.info("Response generated successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error in find_prompt_in_knowledge: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"An error occurred: {error_msg}. Check rag_debug.log for details."


class EvaluationAgent:
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, agent_to_evaluate, max_interactions=5):
        # Initialize the EvaluationAgent with given attributes.
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = agent_to_evaluate
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        final_response = None
        final_evaluation = None
        meets_criteria = False

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")
            final_response = response_from_worker

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {self.persona}, an evaluation agent."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")
            final_evaluation = evaluation

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                meets_criteria = True
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are {self.persona}, providing correction instructions."},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        
        return {
            "final_response": final_response,
            "evaluation": final_evaluation,
            "iterations": i + 1,
            "meets_criteria": meets_criteria
        }



class RoutingAgent:
    """
    A routing agent that directs user prompts to the most appropriate specialized agent.
    Uses embedding similarity to find the best match between the prompt and agent descriptions.
    """
    def __init__(self, openai_api_key, agents):
        """
        Initialize the routing agent.
        
        Parameters:
        openai_api_key (str): API key for OpenAI services
        agents (list): List of agent configurations, each containing 'name', 'description', and 'func'
        """
        self.openai_api_key = openai_api_key
        self.agents = agents  # Store the list of agent configurations

    def get_embedding(self, text):
        """
        Calculate the embedding vector for a given text.
        
        Parameters:
        text (str): The text to embed
        
        Returns:
        list: The embedding vector
        """
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        embedding = response.data[0].embedding
        return embedding 

    def route(self, user_input):
        """
        Route the user prompt to the most appropriate agent based on embedding similarity.
        
        Parameters:
        user_input (str): The user's query or prompt
        
        Returns:
        str: The response from the selected agent
        """
        # Compute the embedding of the user input
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            # Calculate cosine similarity between prompt and agent description
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(f"Similarity with {agent['name']}: {similarity:.3f}")

            # Select the agent with the highest similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        # Handle the case when no suitable agent is found
        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        # Log the selected agent and call its function with the user input
        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)




class ActionPlanningAgent:
    """
    An agent that extracts actionable steps from a user prompt based on predefined knowledge.
    The agent breaks down complex tasks into logical, sequential steps for implementation.
    """

    def __init__(self, openai_api_key, knowledge):
        """
        Initialize the ActionPlanningAgent with an API key and domain knowledge.
        
        Parameters:
        openai_api_key (str): API key for accessing OpenAI services
        knowledge (str): Domain-specific knowledge to guide the planning process
        """
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        """
        Extract actionable steps from a user prompt using the agent's knowledge.
        
        Parameters:
        prompt (str): The user's request or task description
        
        Returns:
        list: A list of actionable steps to complete the requested task
        
        Raises:
        Exception: If there's an error communicating with the OpenAI API
        """
        try:
            # Instantiate the OpenAI client
            client = OpenAI(api_key=self.openai_api_key)
            
            # Prepare system prompt with agent's knowledge
            system_prompt = f"""
            You are an action planning agent. Using your knowledge, you extract from the user prompt 
            the steps requested to complete the action the user is asking for. You return the steps 
            as a numbered list. Only return steps based on your knowledge. Forget any previous context.
            
            This is your knowledge:
            {self.knowledge}
            """
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Extract and process the response text
            response_text = response.choices[0].message.content.strip()

            # Clean and format the extracted steps
            # Remove numbering, bullet points, and other formatting
            clean_steps = []
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Remove numbering (e.g., "1.", "Step 1:", etc.)
                line = re.sub(r'^(\d+\.|\*|Step \d+:|-)\s*', '', line)
                if line:
                    clean_steps.append(line)
            
            return clean_steps
            
        except Exception as e:
            print(f"Error extracting steps: {str(e)}")
            return ["Error: Unable to extract steps from the prompt."]
