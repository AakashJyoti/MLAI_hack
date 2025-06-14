import os
import tiktoken
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryAnswerType, QueryCaptionType, QueryType
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
import asyncio
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import time

from .tools import create_tools
from .prompts import generate_method_prompt, rag_prompt, sales_pitch_prompt
from .agent_manager import create_agent, create_agent_executor, MEMORY_KEY, create_sales_pitch_agent

load_dotenv()

class ABHFL:
    is_function_calling = 0
    is_sales_pitch_active = False
    is_rag_function_active = False

    def __init__(self, message):
        self.API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.Completion_Model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.client = AzureChatOpenAI(
            api_key=self.API_KEY,
            api_version="2023-07-01-preview",
            azure_endpoint=self.RESOURCE_ENDPOINT,
            azure_deployment=self.Completion_Model,
        )
        self.folder_path = "Prompts"
        self.message = message
        self.AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT")
        self.AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
        self.AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
        self.ENCODING = "cl100k_base"
        self.search_client = SearchClient(
            endpoint=self.AZURE_COGNITIVE_SEARCH_ENDPOINT,
            index_name=self.AZURE_COGNITIVE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(self.AZURE_COGNITIVE_SEARCH_API_KEY),
        )
        self.user_input = ""
        self.store = {}
        self.encoding = tiktoken.encoding_for_model("gpt-4-0613")

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        return len(self.encoding.encode(string))

    def reset_system_message(self):
        """Reset SystemMessage to the original main2.txt prompt."""
        if not ABHFL.is_sales_pitch_active:
            with open("prompts/main_prompt2.txt", "r", encoding="utf-8") as f:
                text = f.read()
            self.message = [SystemMessage(content=f"{text}")]

    def append_to_system_message(self, content):
        """Append content to the system message."""
        if self.message and isinstance(self.message[0], SystemMessage):
            self.message[0].content += f"\n{content}"
        else:
            self.message.insert(0, SystemMessage(content=content))

    # def generate_method(self, prompt_name):
    #     """Generic method to handle various prompts and update system message."""
    #     prompt = generate_method_prompt(prompt_name, self.user_input)
    #     if prompt:
    #         replaced = False
    #         for i, message in enumerate(self.message):
    #             if isinstance(message, SystemMessage):
    #                 self.message[i] = SystemMessage(content=prompt)
    #                 replaced = True
    #                 break
    #         if not replaced:
    #             self.message.append(SystemMessage(content=prompt))
    #         return prompt
    #     return None

    # def collateral_type(self):
    #     return self.generate_method("Collateral")

    # ... other generate_method wrappers ...

    def all_other_information(self, *args, **kwargs):
        """Function provides all details for products using RAG."""
        if not ABHFL.is_rag_function_active:
            ABHFL.is_rag_function_active = True
            print("Rag function called")
            question = self.user_input
            max_tokens = 6000
            token_threshold = 0.8 * max_tokens
            results = self.search_client.search(
                search_text=question,
                select=["product", "description"],
                top=3,
            )

            context = ""
            total_tokens = 0

            for result in results:
                title = result["product"]
                content = result["description"]
                result_tokens = self.num_tokens_from_string(content)
                if total_tokens + result_tokens > token_threshold:
                    break
                context += f"{{'Title': {title} , 'Product Details': {content} }}\n "
                total_tokens += result_tokens

            prompt = rag_prompt(context, question)
            if prompt:
                replaced = False
                for i, message in enumerate(self.message):
                    if isinstance(message, SystemMessage):
                        self.message[i] = SystemMessage(content=prompt)
                        replaced = True
                        break
                if not replaced:
                    self.message.append(SystemMessage(content=prompt))
                ABHFL.is_function_calling = 12

    def generate_salespitch(self, *args, **kwargs):
        """Generate a sales pitch."""
        if not ABHFL.is_sales_pitch_active:
            ABHFL.is_sales_pitch_active = True
            print("Sales pitch function called")
            sales_pitch = sales_pitch_prompt()
            if sales_pitch:
                replaced = False
                for i, message in enumerate(self.message):
                    if isinstance(message, SystemMessage):
                        self.message[i] = SystemMessage(content=sales_pitch)
                        replaced = True
                        break
                if not replaced:
                    self.message.append(SystemMessage(content=sales_pitch))
                self.message.append(HumanMessage(content=self.user_input))
                ABHFL.is_function_calling = 11

    async def run_conversation(self, user_input):
        """Run the conversation with the agent."""
        self.user_input = user_input
        self.message.append(HumanMessage(content=user_input))

        # Create tools and agent
        tools = create_tools(self)
        agent = create_agent(self.client, tools)
        agent_executor = create_agent_executor(agent, tools)

        max_response_tokens = 250
        token_limit = 50000

        def calculate_token_length(messages):
            num_tokens = 0
            for message in messages:
                num_tokens += 3  # tokens_per_message
                num_tokens += len(self.encoding.encode(message.content))
            num_tokens += 3  # every reply is primed
            return num_tokens

        def ensure_message_length_within_limit():
            conv_history_tokens = calculate_token_length(self.message)
            while conv_history_tokens + max_response_tokens >= token_limit:
                if len(self.message) > 1:
                    del self.message[1]
                    conv_history_tokens = calculate_token_length(self.message)
                else:
                    break

        with get_openai_callback() as cb:
            try:
                ensure_message_length_within_limit()
                async for chunk in agent_executor.astream_events(
                    {"input": user_input, "chat_history": self.message}, version="v1"
                ):
                    time.sleep(0.05)
                    yield chunk
            except Exception as e:
                error_message = f"An error occurred: {e}"
                print(error_message)
                yield {"error": error_message}
