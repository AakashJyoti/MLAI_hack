# import streamlit as st
# import pandas as pd
# import json
# import openai
# import os
# import re
# import matplotlib.pyplot as plt
# # from langchain_experimental.agents.create_csv_agent import create_csv_agent
# from langchain_experimental.agents import create_csv_agent
# from langchain_openai import AzureChatOpenAI
# from langchain.agents.agent_types import AgentType

# # Load the OpenAI API key

# API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
#         # self.client = AzureOpenAI(api_key=self.API_KEY, api_version="2023-07-01-preview",
#         #                           azure_endpoint=self.RESOURCE_ENDPOINT)
# Completion_Model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
# client = AzureChatOpenAI(
#             api_key=API_KEY,
#             api_version="2023-07-01-preview",
#             azure_endpoint=RESOURCE_ENDPOINT,
#             azure_deployment=Completion_Model,
#         )

# def csv_agent_func(file_path, user_message):
#     """Run the CSV agent with the given file path and user message."""
#     agent = create_csv_agent(
#         client,
#         file_path, 
#         verbose=True,
#         agent_type=AgentType.OPENAI_FUNCTIONS,
#         allow_dangerous_code=True
#     )

#     try:
#         # Properly format the user's input and wrap it with the required "input" key
#         tool_input = {
#             "input": {
#                 "name": "python",
#                 "arguments": user_message
#             }
#         }
        
#         response = agent.invoke(tool_input)
#         return response
#     except Exception as e:
#         st.write(f"Error: {e}")
#         return None

# def display_content_from_json(json_response):
#     """
#     Display content to Streamlit based on the structure of the provided JSON.
#     """
    
#     # Check if the response has plain text.
#     if "answer" in json_response:
#         st.write(json_response["answer"])

#     # Check if the response has a bar chart.
#     if "bar" in json_response:
#         data = json_response["bar"]
#         df = pd.DataFrame(data)
#         df.set_index("columns", inplace=True)
#         st.bar_chart(df)

#     # Check if the response has a table.
#     if "table" in json_response:
#         data = json_response["table"]
#         df = pd.DataFrame(data["data"], columns=data["columns"])
#         st.table(df)

# def extract_code_from_response(response):
#     """Extracts Python code from a string response."""
#     # Use a regex pattern to match content between triple backticks
#     code_pattern = r"```python(.*?)```"
#     match = re.search(code_pattern, response, re.DOTALL)
    
#     if match:
#         # Extract the matched code and strip any leading/trailing whitespaces
#         return match.group(1).strip()
#     return None


# def csv_analyzer_app():
#     """Main Streamlit application for CSV analysis."""

#     st.title('CSV Assistant')
#     st.write('Please upload your CSV file and enter your query below:')
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
#     if uploaded_file is not None:
#         file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
#         st.write(file_details)
        
#         # Save the uploaded file to disk
#         file_path = os.path.join("tmp", uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
        
#         df = pd.read_csv(file_path)
#         st.dataframe(df)
        
#         user_input = st.text_input("Your query")
#         if st.button('Run'):
#             response = csv_agent_func(file_path, user_input)
#             print(response)
#             # Extracting code from the response
#             code_to_execute = extract_code_from_response(response)
#             print(code_to_execute)
#             if code_to_execute:
#                 try:
#                     # Making df available for execution in the context
#                     exec(code_to_execute, globals(), {"df": df, "plt": plt})
#                     fig = plt.gcf()  # Get current figure
#                     st.pyplot(fig)  # Display using Streamlit
#                 except Exception as e:
#                     st.write(f"Error executing code: {e}")
#             else:
#                 st.write(response)

#     st.divider()

# if __name__ == "__main__":
#     csv_analyzer_app()

import pandas as pd

def filter_csv( hfc, state, district=None, pincode=None):
    file_path = "HFC Competitors - Product Benchmarking.csv"
    # Load the CSV into a DataFrame
    data = pd.read_csv(file_path)
    
    # Filter based on hfc and state (mandatory)
    filtered_data = data[(data['hfc'] == hfc.lower()) & (data['state'] == state.lower())]
    
    # Apply optional filters for district and pincode
    if district:
        filtered_data = filtered_data[filtered_data['district'] == district.lower()]
    if pincode:
        filtered_data = filtered_data[filtered_data['pincode'] == pincode]
    
    return filtered_data

# Example usage

# result = filter_csv(hfc="iifl", state="maharashtra")
# print(result)
