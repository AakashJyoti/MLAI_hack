import pandas as pd
import openai
import os
import re
from langchain_experimental.agents import create_csv_agent
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType

# Azure OpenAI Configuration
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
Completion_Model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Initialize Azure Chat OpenAI Client
client = AzureChatOpenAI(
    api_key=API_KEY,
    api_version="2023-07-01-preview",
    azure_endpoint=RESOURCE_ENDPOINT,
    azure_deployment=Completion_Model,
)

csv_file_path = "HFC Competitors - Product Benchmarking.csv"

def query_csv(user_query):
    """
    Process a query using the predefined CSV file and return the response.

    Args:
        file_path (str): Path to the predefined CSV file.
        user_query (str): Query to analyze the CSV file.

    Returns:
        dict: Response from the agent or error message.
    """
    try:
        # Create the CSV Agent
        agent = create_csv_agent(
            llm=client,
            path=csv_file_path,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code=True,
        )

        # Invoke the agent with the user query
        tool_input = {"input": {"name": "python", "arguments": user_query}}
        response = agent.invoke(tool_input)

        return response
    except Exception as e:
        return {"error": str(e)}

# # Example Usage
# if __name__ == "__main__":
#     # Define the path to your CSV file
    
#     query = "Provide the summary of the dataset."
    
#     # Call the query function
#     result = query_csv(query)
    
#     # Print the result
#     print(result)
