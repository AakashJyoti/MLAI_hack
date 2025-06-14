import os
import tiktoken
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryAnswerType, QueryCaptionType, QueryType
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import asyncio
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor
from langchain_community.callbacks import get_openai_callback
import numpy_financial as npf
import pandas as pd
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent , ConversationalAgent
import time
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
import base64

import httpx
load_dotenv()


class ABHFL_FILES:
    is_function_calling = 0
    is_sales_pitch_active = False
    is_rag_function_active = False

    def __init__(self, message):
        self.API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        print(self.API_KEY)
        self.RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        # self.client = AzureOpenAI(api_key=self.API_KEY, api_version="2023-07-01-preview",
        #                           azure_endpoint=self.RESOURCE_ENDPOINT)
        self.Completion_Model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.client = AzureChatOpenAI(
            api_key=self.API_KEY,
            api_version="2023-07-01-preview",
            azure_endpoint=self.RESOURCE_ENDPOINT,
            azure_deployment=self.Completion_Model,
        )
        self.folder_path = "Prompts"
        self.message = message
        self.AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv(
            "AZURE_COGNITIVE_SEARCH_ENDPOINT"
        )
        self.AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv(
            "AZURE_COGNITIVE_SEARCH_API_KEY"
        )
        self.AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv(
            "AZURE_COGNITIVE_SEARCH_INDEX_NAME"
        )
        self.ENCODING = "cl100k_base"
        self.search_client = SearchClient(
            endpoint=self.AZURE_COGNITIVE_SEARCH_ENDPOINT,
            index_name=self.AZURE_COGNITIVE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(self.AZURE_COGNITIVE_SEARCH_API_KEY),
        )

        self.user_input = ""
        self.store = {}

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    async def run_conversation(self, user_input , image_data):
        self.user_input = user_input
        self.message.append(HumanMessage(content=[
        {"type": "text", "text": "describe the this image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
))
        
        MEMORY_KEY = "chat_history"
        prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", """You are a key figure at Aditya Birla Housing Finance Limited (ABHFL), but you have only limited information about the company. """),
                (
                    "system",
                    """You are an expert conversational sales manager with access to various tools to deliver clear and concise answers.avoiding unnecessary details. When responding to general or open-ended questions, always leverage tools for accuracy. If unsure of an answer, ask follow-up questions to clarify. You are experienced and professional in this role.""",
                ),
                MessagesPlaceholder(variable_name=MEMORY_KEY)
            ]
        )

        # prompt_value = prompt.format_prompt(input=user_input, chat_history=self.message)  # Include chat_history
        # messages = prompt_value.to_messages()  # Convert PromptValue to list of BaseMessages
        # chat1 = await self.client.ainvoke(messages)
        parser = StrOutputParser()
        chain = prompt | self.client | parser  # Pass the list of BaseMessages
        max_response_tokens = 250
        token_limit = 50000

        # Helper function to calculate total tokens in the messages
        def calculate_token_length(messages):
            tokens_per_message = 3
            tokens_per_name = 1
            # encoding = tiktoken.get_encoding(self.ENCODING)
            encoding = tiktoken.encoding_for_model("gpt-4-0613")
            num_tokens = 0
            # print(len(messages))
            for message in messages:
                # print(message.content)
                num_tokens += tokens_per_message
                # for value in message.content:
                #     print(value)
                num_tokens += len(encoding.encode(message.content))
                # if key == "name":
                #     num_tokens += tokens_per_name
            num_tokens += 3
            # print(num_tokens)  # every reply is primed with <|start|>assistant<|message|>
            return num_tokens

        # Helper function to ensure message length is within limits
        def ensure_message_length_within_limit(message):
            # print("Lenth Function Called",messages[0])
            messages = self.message
            conv_history_tokens = calculate_token_length(self.message)

            while conv_history_tokens + max_response_tokens >= token_limit:
                # print("Conv History", conv_history_tokens)
                if len(self.message) > 1:
                    del self.message[1]  # Remove the oldest message
                    conv_history_tokens = calculate_token_length(self.message)

        with get_openai_callback() as cb:
            # print(self.message)
            try:
                # print(self.message)
                # ensure_message_length_within_limit(
                #     self.message
                # )  # Reserve some tokens for functions and overhead
                
                async for chunk in chain.astream(
                    {"input": user_input, "chat_history": self.message}, version="v1"
                ):
                    time.sleep(0.05)
                    # print("|")
                    yield chunk # Stream the response content
            except Exception as e:
                error_message = f"An error occurred: {e}"
                print(error_message)
                yield error_message
            # print("Token : ", cb)


# Create a single event loop for all requests
global_event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(global_event_loop)

# Utility function to iterate over async generator
def iter_over_async(ait):
    ait = ait.__aiter__()
    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None
        # Use the global event loop
    global global_event_loop
    loop = global_event_loop

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj

# run above code
abhfl = ABHFL_FILES([])
if not abhfl.message:
                abhfl.message.append(SystemMessage(content="you are a great conversational AI"))
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

for i in iter_over_async(abhfl.run_conversation("describe the weather in this image?",image_data)):
    print(i, end="|", flush=True)



