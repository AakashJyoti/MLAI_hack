import os
from dotenv import load_dotenv
import boto3
import anthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.llms.bedrock import Bedrock
from langchain_community.retrievers import AmazonKendraRetriever
import asyncio
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import time

from .tools import create_tools
from .prompts import generate_method_prompt, rag_prompt, sales_pitch_prompt
from .agent_manager import (
    create_agent,
    create_agent_executor,
    MEMORY_KEY,
    create_sales_pitch_agent,
)

load_dotenv()


class ABHFL:
    is_function_calling = 0
    is_sales_pitch_active = False
    is_rag_function_active = False

    def __init__(self, message):
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION = os.getenv("AWS_REGION")
        self.BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
        self.BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS"))
        self.BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE"))
        self.KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
        self.KENDRA_MIN_CONFIDENCE_SCORE = float(
            os.getenv("KENDRA_MIN_CONFIDENCE_SCORE")
        )

        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.AWS_REGION,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
        )

        # Initialize Bedrock LLM
        self.client = Bedrock(
            client=self.bedrock_client,
            model_id=self.BEDROCK_MODEL_ID,
            model_kwargs={
                "maxTokens": self.BEDROCK_MAX_TOKENS,
                "temperature": self.BEDROCK_TEMPERATURE,
            },
        )

        # Initialize Kendra client
        self.kendra_client = boto3.client(
            service_name="kendra",
            region_name=self.AWS_REGION,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
        )

        # Initialize Kendra retriever
        self.retriever = AmazonKendraRetriever(
            index_id=self.KENDRA_INDEX_ID,
            region_name=self.AWS_REGION,
            credentials_profile_name=None,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            top_k=3,
            attribute_filter=None,
        )

        self.folder_path = "Prompts"
        self.message = message
        self.user_input = ""
        self.store = {}
        # Initialize Claude tokenizer for accurate token counting
        self.cl = anthropic.Anthropic()

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string using Claude's tokenizer."""
        try:
            return self.cl.count_tokens(string)
        except Exception as e:
            # Fallback method: approximate tokens (4 characters per token on average)
            return len(string) // 4

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

    def all_other_information(self, *args, **kwargs):
        """Function provides all details for products using RAG with AWS Kendra."""
        if not ABHFL.is_rag_function_active:
            ABHFL.is_rag_function_active = True
            print("Rag function called")
            question = self.user_input
            max_tokens = 6000
            token_threshold = 0.8 * max_tokens

            # Use Kendra retriever to get relevant documents
            results = self.retriever.get_relevant_documents(question)

            context = ""
            total_tokens = 0

            for doc in results:
                if hasattr(doc.metadata, "title"):
                    title = doc.metadata["title"]
                else:
                    title = "Document"
                content = doc.page_content
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
