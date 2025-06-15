import os
import json
import boto3
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_aws import ChatBedrock
from langchain_community.callbacks import get_openai_callback
import asyncio
from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
import time
from typing import Dict, List, Any, AsyncGenerator, Union

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
        # AWS Configuration
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
        self.BEDROCK_MODEL_ID = os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        self.BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
        self.BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.7"))
        self.KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
        self.KENDRA_MIN_CONFIDENCE_SCORE = float(
            os.getenv("KENDRA_MIN_CONFIDENCE_SCORE", "0.7")
        )

        # Initialize AWS clients
        self.session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_REGION,
        )

        # Initialize Bedrock client for Claude
        self.bedrock_client = self.session.client("bedrock-runtime")

        # Initialize Kendra client for search
        self.kendra_client = self.session.client("kendra")

        # Initialize LangChain Claude client
        self.client = ChatBedrock(
            client=self.bedrock_client,
            model_id=self.BEDROCK_MODEL_ID,
            model_kwargs={
                "max_tokens": self.BEDROCK_MAX_TOKENS,
                "temperature": self.BEDROCK_TEMPERATURE,
            },
        )

        self.folder_path = "Prompts"
        self.message = message
        self.user_input = ""
        self.store = {}

    def num_tokens_from_string(self, string: str) -> int:
        """
        Approximate token count for Claude models.
        Claude uses a different tokenizer than GPT, so this is an approximation.
        """
        # Rough approximation: 1 token ≈ 4 characters for English text
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
            # Ensure system message is always first
            self.message.insert(0, SystemMessage(content=content))

    def search_kendra(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search using AWS Kendra instead of Azure Cognitive Search.
        """
        try:
            response = self.kendra_client.query(
                IndexId=self.KENDRA_INDEX_ID,
                QueryText=query,
                PageSize=max_results,
                AttributeFilter={
                    "EqualsTo": {
                        "Key": "_language_code",
                        "Value": {"StringValue": "en"},
                    }
                },
            )

            results = []
            for item in response.get("ResultItems", []):
                # Filter by confidence score
                if (
                    item.get("ScoreAttributes", {}).get("ScoreConfidence", "LOW")
                    == "HIGH"
                    or item.get("ScoreAttributes", {}).get("ScoreConfidence", "LOW")
                    == "MEDIUM"
                ):

                    # Extract document attributes
                    title = item.get("DocumentTitle", {}).get("Text", "Unknown")
                    content = item.get("DocumentExcerpt", {}).get("Text", "")

                    # Try to get additional attributes if available
                    attributes = item.get("DocumentAttributes", [])
                    product_info = {}

                    for attr in attributes:
                        key = attr.get("Key", "")
                        value = attr.get("Value", {})
                        if "StringValue" in value:
                            product_info[key] = value["StringValue"]
                        elif "StringListValue" in value:
                            product_info[key] = value["StringListValue"]

                    results.append(
                        {
                            "product": title,
                            "description": content,
                            "attributes": product_info,
                        }
                    )

            return results

        except Exception as e:
            print(f"Error searching Kendra: {e}")
            return []

    def all_other_information(self, *args, **kwargs):
        """Function provides all details for products using RAG with Kendra."""
        if not ABHFL.is_rag_function_active:
            ABHFL.is_rag_function_active = True
            print("RAG function called")
            question = self.user_input
            max_tokens = 6000
            token_threshold = 0.8 * max_tokens

            # Use Kendra instead of Azure Cognitive Search
            results = self.search_kendra(question, max_results=3)

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
                # Always replace or create system message at the beginning
                self.ensure_system_message_first(prompt)
                ABHFL.is_function_calling = 12

    def generate_salespitch(self, *args, **kwargs):
        """Generate a sales pitch."""
        if not ABHFL.is_sales_pitch_active:
            ABHFL.is_sales_pitch_active = True
            print("Sales pitch function called")
            sales_pitch = sales_pitch_prompt()
            if sales_pitch:
                # Always replace or create system message at the beginning
                self.ensure_system_message_first(sales_pitch)
                self.message.append(HumanMessage(content=self.user_input))
                ABHFL.is_function_calling = 11

    def ensure_system_message_first(self, content):
        """Ensure system message is always at the beginning of message list."""
        # Remove any existing system messages
        self.message = [
            msg for msg in self.message if not isinstance(msg, SystemMessage)
        ]
        # Add new system message at the beginning
        self.message.insert(0, SystemMessage(content=content))

    def clean_message_order(self):
        """Clean and reorder messages to ensure Claude compatibility."""
        system_messages = []
        other_messages = []

        for msg in self.message:
            if isinstance(msg, SystemMessage):
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Combine all system messages into one
        if system_messages:
            combined_system_content = "\n\n".join(
                [msg.content for msg in system_messages]
            )
            self.message = [
                SystemMessage(content=combined_system_content)
            ] + other_messages
        else:
            self.message = other_messages
        """
        Direct invocation of Claude via Bedrock API for more control.
        """
        try:
            # Prepare the request body for Claude
            system_message = ""
            user_messages = []

            for msg in messages:
                if msg.get("role") == "system":
                    system_message = msg.get("content", "")
                else:
                    user_messages.append(msg)

            # Claude API request format
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.BEDROCK_MAX_TOKENS,
                "temperature": self.BEDROCK_TEMPERATURE,
                "messages": user_messages,
            }

            if system_message:
                request_body["system"] = system_message

            response = self.bedrock_client.invoke_model(
                modelId=self.BEDROCK_MODEL_ID, body=json.dumps(request_body)
            )

            response_body = json.loads(response["body"].read())
            return response_body.get("content", [{}])[0].get("text", "")

        except Exception as e:
            print(f"Error invoking Claude: {e}")
            return f"Error: {str(e)}"

    async def run_conversation(
        self, user_input: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the conversation with the agent using Claude."""
        self.user_input = user_input
        self.message.append(HumanMessage(content=user_input))

        # Clean message order before processing
        self.clean_message_order()

        # Create tools and agent
        tools = create_tools(self)
        agent = create_agent(self.client, tools)
        agent_executor = create_agent_executor(agent, tools)

        max_response_tokens = 250
        token_limit = 50000  # Adjust based on Claude's context window

        def calculate_token_length(messages):
            """Calculate approximate token length for Claude."""
            num_tokens = 0
            for message in messages:
                num_tokens += 3  # tokens_per_message
                num_tokens += self.num_tokens_from_string(message.content)
            num_tokens += 3  # every reply is primed
            return num_tokens

        def ensure_message_length_within_limit():
            """Ensure conversation history fits within token limits."""
            conv_history_tokens = calculate_token_length(self.message)
            while conv_history_tokens + max_response_tokens >= token_limit:
                if len(self.message) > 1:
                    # Always keep system message (first), remove second message
                    if len(self.message) > 2:
                        del self.message[1]
                    else:
                        break
                    conv_history_tokens = calculate_token_length(self.message)
                else:
                    break

        # Note: get_openai_callback might not work with Claude, so we'll handle it gracefully
        try:
            with get_openai_callback() as cb:
                try:
                    ensure_message_length_within_limit()
                    # Clean message order one more time before execution
                    self.clean_message_order()
                    async for chunk in agent_executor.astream_events(
                        {"input": user_input, "chat_history": self.message},
                        version="v1",
                    ):
                        time.sleep(0.05)
                        yield chunk
                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    print(error_message)
                    yield {"error": error_message}
        except Exception as callback_error:
            # If callback doesn't work with Claude, run without it
            print(f"Callback not supported: {callback_error}")
            try:
                ensure_message_length_within_limit()
                # Clean message order one more time before execution
                self.clean_message_order()
                async for chunk in agent_executor.astream_events(
                    {"input": user_input, "chat_history": self.message}, version="v1"
                ):
                    time.sleep(0.05)
                    yield chunk
            except Exception as e:
                error_message = f"An error occurred: {e}"
                print(error_message)
                yield {"error": error_message}

    def get_bedrock_models(self):
        """Get available Bedrock models for reference."""
        try:
            bedrock_client = self.session.client("bedrock")
            response = bedrock_client.list_foundation_models()
            return [
                model["modelId"]
                for model in response["modelSummaries"]
                if "claude" in model["modelId"].lower()
            ]
        except Exception as e:
            print(f"Error getting Bedrock models: {e}")
            return []

    def test_kendra_connection(self):
        """Test Kendra connection and configuration."""
        try:
            response = self.kendra_client.describe_index(Id=self.KENDRA_INDEX_ID)
            print(f"Kendra Index Status: {response['Status']}")
            return True
        except Exception as e:
            print(f"Error connecting to Kendra: {e}")
            return False
