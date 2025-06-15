import os
import time
import logging
from typing import List, Dict, Any, AsyncGenerator
from dotenv import load_dotenv
import boto3
import anthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKendraRetriever
import asyncio
from langchain.agents import AgentExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent

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
    """
    Advanced Bot Handler for Human-like Feedback (ABHFL) class that manages conversations
    with AI agents, including RAG capabilities and sales pitch generation.

    Attributes:
        is_function_calling (int): Flag indicating active function call
        is_sales_pitch_active (bool): Flag for sales pitch mode
        is_rag_function_active (bool): Flag for RAG mode
    """

    def __init__(self, message: List[Any]):
        """
        Initialize ABHFL with configuration and clients.

        Args:
            message: Initial message list containing SystemMessage
        """
        self.logger = logging.getLogger(__name__)

        # Validate and load configuration
        self._load_configuration()
        self._validate_configuration()

        # Initialize clients
        self._initialize_clients()

        # Initialize state
        self.message = message
        self.user_input = ""
        self.store = {}
        self.is_function_calling = 0
        self.is_sales_pitch_active = False
        self.is_rag_function_active = False

        # Initialize Claude tokenizer
        self.cl = anthropic.Anthropic()

    def _load_configuration(self):
        """Load configuration from environment variables."""
        try:
            self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
            self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
            self.AWS_REGION = os.getenv("AWS_REGION")
            self.BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
            self.BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "1024"))
            self.BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.5"))
            self.KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
            self.KENDRA_MIN_CONFIDENCE_SCORE = float(
                os.getenv("KENDRA_MIN_CONFIDENCE_SCORE", "0.75")
            )
        except (ValueError, TypeError) as e:
            self.logger.error(f"Configuration error: {e}")
            raise ValueError("Invalid configuration values") from e

    def _validate_configuration(self):
        """Validate required configuration."""
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "BEDROCK_MODEL_ID",
            "KENDRA_INDEX_ID",
        ]

        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required configuration: {missing_vars}")

    def _initialize_clients(self):
        """Initialize AWS clients."""
        try:
            # Initialize Bedrock client
            self.bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.AWS_REGION,
                aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            )

            # Initialize ChatBedrock LLM
            self.client = ChatBedrock(
                client=self.bedrock_client,
                model_id=self.BEDROCK_MODEL_ID,
                model_kwargs={
                    "max_tokens": self.BEDROCK_MAX_TOKENS,
                    "temperature": self.BEDROCK_TEMPERATURE,
                },
            )

            # Initialize Kendra client first
            kendra_client = boto3.client(
                "kendra",
                region_name=self.AWS_REGION,
                aws_access_key_id=self.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            )

            # Initialize Kendra retriever with explicit client
            self.retriever = AmazonKendraRetriever(
                index_id=self.KENDRA_INDEX_ID,
                client=kendra_client,
                top_k=3,
                attribute_filter={
                    "AndAllFilters": [
                        {
                            "GreaterThanOrEquals": {
                                "Key": "_confidence_score",
                                "Value": self.KENDRA_MIN_CONFIDENCE_SCORE,
                            }
                        }
                    ]
                },
            )
        except Exception as e:
            self.logger.error(f"Client initialization failed: {e}")
            raise

    def num_tokens_from_string(self, string: str) -> int:
        """
        Calculate the number of tokens in a text string.

        Args:
            string: Input text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            return self.cl.count_tokens(string)
        except Exception as e:
            self.logger.warning(f"Token count failed, using fallback: {e}")
            return len(string) // 4  # Fallback approximation

    def reset_system_message(self):
        """Reset SystemMessage to the original main2.txt prompt."""
        if not self.is_sales_pitch_active:
            try:
                with open("prompts/main_prompt2.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                self._replace_or_append_system_message(text)
            except Exception as e:
                self.logger.error(f"Failed to reset system message: {e}")
                raise

    def _replace_or_append_system_message(self, content: str):
        """
        Helper method to replace or append system message content.

        Args:
            content: New content for the system message
        """
        for i, message in enumerate(self.message):
            if isinstance(message, SystemMessage):
                self.message[i] = SystemMessage(content=content)
                return
        self.message.insert(0, SystemMessage(content=content))

    def append_to_system_message(self, content: str):
        """Append content to the system message."""
        if self.message and isinstance(self.message[0], SystemMessage):
            self.message[0].content += f"\n{content}"
        else:
            self.message.insert(0, SystemMessage(content=content))

    def all_other_information(self, *args, **kwargs):
        """Retrieve product details using RAG with AWS Kendra."""
        if not self.is_rag_function_active:
            try:
                self.is_rag_function_active = True
                self.logger.info("RAG function called")

                question = self.user_input
                max_tokens = 6000
                token_threshold = 0.8 * max_tokens
                context = ""
                total_tokens = 0

                # Retrieve relevant documents
                results = self.retriever.get_relevant_documents(question)

                for doc in results:
                    title = getattr(doc, "metadata", {}).get("title", "Document")
                    content = doc.page_content
                    result_tokens = self.num_tokens_from_string(content)

                    if total_tokens + result_tokens > token_threshold:
                        break

                    context += f"{{'Title': {title}, 'Product Details': {content}}}\n"
                    total_tokens += result_tokens

                prompt = rag_prompt(context, question)
                if prompt:
                    self._replace_or_append_system_message(prompt)
                    self.is_function_calling = 12
            except Exception as e:
                self.logger.error(f"RAG function failed: {e}")
                self.is_rag_function_active = False
                raise

    def generate_salespitch(self, *args, **kwargs):
        """Generate a sales pitch."""
        if not self.is_sales_pitch_active:
            try:
                self.is_sales_pitch_active = True
                self.logger.info("Sales pitch function called")

                sales_pitch = sales_pitch_prompt()
                if sales_pitch:
                    self._replace_or_append_system_message(sales_pitch)
                    self.message.append(HumanMessage(content=self.user_input))
                    self.is_function_calling = 11
            except Exception as e:
                self.logger.error(f"Sales pitch generation failed: {e}")
                self.is_sales_pitch_active = False
                raise

    async def cleanup(self):
        """Clean up resources."""
        try:
            # Add any cleanup logic for clients if needed
            pass
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    async def run_conversation(
        self, user_input: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the conversation with the agent.

        Args:
            user_input: User's input message

        Yields:
            Dictionary with conversation events or errors
        """
        if not user_input or not isinstance(user_input, str):
            yield {"error": "user_input must be a non-empty string"}
            return

        self.user_input = user_input
        self.message.append(HumanMessage(content=user_input))

        try:
            tools = create_tools(self)
            agent = create_agent(self.client, tools)
            agent_executor = create_agent_executor(agent, tools)

            self._ensure_message_length_within_limit()

            async for chunk in agent_executor.astream_events(
                {"input": user_input, "chat_history": self.message}, version="v1"
            ):
                await asyncio.sleep(0.05)  # Better than time.sleep for async
                yield chunk

        except Exception as e:
            error_msg = f"Conversation error: {e}"
            self.logger.error(error_msg)
            yield {"error": error_msg}

    def _ensure_message_length_within_limit(self):
        """Ensure conversation history stays within token limits."""
        max_response_tokens = 250
        token_limit = 50000
        conv_history_tokens = self._calculate_token_length(self.message)

        while conv_history_tokens + max_response_tokens >= token_limit:
            if len(self.message) > 1:
                del self.message[1]
                conv_history_tokens = self._calculate_token_length(self.message)
            else:
                break

    def _calculate_token_length(self, messages: List[Any]) -> int:
        """
        Calculate total tokens in message history.

        Args:
            messages: List of message objects

        Returns:
            Total token count
        """
        num_tokens = 0
        for message in messages:
            num_tokens += 3  # tokens_per_message
            num_tokens += self.num_tokens_from_string(message.content)
        num_tokens += 3  # every reply is primed
        return num_tokens
