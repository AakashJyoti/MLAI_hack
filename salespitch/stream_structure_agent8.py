import os
import tiktoken
from dotenv import load_dotenv
import boto3
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_aws import ChatBedrock , ChatBedrockConverse
from langchain_community.callbacks import get_openai_callback
import asyncio
from langchain.agents import AgentExecutor
import time

from .tools import create_tools
from .prompts import generate_method_prompt, rag_prompt, sales_pitch_prompt
from .agent_manager import (
    create_agent,
    create_agent_executor,
    MEMORY_KEY,
    # create_sales_pitch_agent,
)

load_dotenv()


class ABHFL:
    is_function_calling = 0
    is_sales_pitch_active = False
    is_rag_function_active = False

    def __init__(self, messages):
        # AWS Credentials and Configuration
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION = os.getenv("AWS_REGION")

        # Bedrock Configuration
        self.BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
        self.BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS"))
        self.BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE"))

        # Kendra Configuration
        self.KENDRA_INDEX_ID = os.getenv("KENDRA_INDEX_ID")
        self.KENDRA_MIN_CONFIDENCE_SCORE = float(
            os.getenv("KENDRA_MIN_CONFIDENCE_SCORE")
        )

        # Initialize Bedrock Chat Client
        self.client = ChatBedrock(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_REGION,
            model_id=self.BEDROCK_MODEL_ID,
            beta_use_converse_api=True,
            model_kwargs={
                "temperature": self.BEDROCK_TEMPERATURE,
                "max_tokens_to_sample": self.BEDROCK_MAX_TOKENS,
            },
        )

        # Initialize Kendra Client
        self.kendra_client = boto3.client(
            "kendra",
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            region_name=self.AWS_REGION,
        )

        self.folder_path = "Prompts"
        self.message = messages
        # self.message = message
        self.ENCODING = "cl100k_base"
        self.user_input = ""
        self.store = {}
        self.encoding = tiktoken.get_encoding(self.ENCODING)

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        return len(self.encoding.encode(string))

    # def reset_system_message(self):
    #     """Reset SystemMessage to the original main2.txt prompt."""
    #     if not ABHFL.is_sales_pitch_active:
    #         with open("prompts/main_prompt2.txt", "r", encoding="utf-8") as f:
    #             text = f.read()
    #         self.message = [SystemMessage(content=f"{text}")]

    # def append_to_system_message(self, content):
    #     """Append content to the system message."""
    #     if self.message and isinstance(self.message[0], SystemMessage):
    #         self.message[0].content += f"\n{content}"
    #     else:
    #         self.message.insert(0, SystemMessage(content=content))

    # def all_other_information(self, *args, **kwargs):
    #     """Function provides all details for products using RAG with AWS Kendra."""
    #     if not ABHFL.is_rag_function_active:
    #         ABHFL.is_rag_function_active = True
    #         print("Kendra RAG function called")
    #         question = self.user_input
    #         max_tokens = 6000
    #         token_threshold = 0.8 * max_tokens

    #         try:
    #             response = self.kendra_client.query(
    #                 IndexId=self.KENDRA_INDEX_ID,
    #                 QueryText=question,
    #                 PageSize=3,  # Retrieve top 3 results
    #             )
    #         except Exception as e:
    #             print(f"Error querying Kendra: {e}")
    #             return

    #         context = ""
    #         total_tokens = 0

    #         for result in response.get("ResultItems", []):
    #             confidence = result.get("ScoreAttributes", {}).get("ScoreConfidence")
    #             if (
    #                 confidence
    #                 and confidence.lower() in ["very_high", "high"]
    #                 or (
    #                     not confidence
    #                     and result.get("Score", 0) >= self.KENDRA_MIN_CONFIDENCE_SCORE
    #                 )
    #             ):
    #                 title = result.get("DocumentTitle", {}).get("Text", "No Title")
    #                 content = result.get("DocumentExcerpt", {}).get("Text", "")
    #                 result_tokens = self.num_tokens_from_string(content)

    #                 if total_tokens + result_tokens > token_threshold:
    #                     break

    #                 context += (
    #                     f"{{'Title': {title} , 'Product Details': {content} }}\n "
    #                 )
    #                 total_tokens += result_tokens

            # prompt = rag_prompt(context, question)
            # if prompt:
            #     replaced = False
            #     for i, message in enumerate(self.message):
            #         if isinstance(message, SystemMessage):
            #             self.message[i] = SystemMessage(content=prompt)
            #             replaced = True
            #             break
            #     if not replaced:
            #         self.message.append(SystemMessage(content=prompt))
            #     ABHFL.is_function_calling = 12

    # def generate_salespitch(self, *args, **kwargs):
    #     """Generate a sales pitch."""
    #     if not ABHFL.is_sales_pitch_active:
    #         ABHFL.is_sales_pitch_active = True
    #         print("Sales pitch function called")
    #         sales_pitch = sales_pitch_prompt()
    #         if sales_pitch:
    #             replaced = False
    #             for i, message in enumerate(self.message):
    #                 if isinstance(message, SystemMessage):
    #                     self.message[i] = SystemMessage(content=sales_pitch)
    #                     replaced = True
    #                     break
    #             if not replaced:
    #                 self.message.append(SystemMessage(content=sales_pitch))
    #             self.message.append(HumanMessage(content=self.user_input))
    #             ABHFL.is_function_calling = 11

    def run_conversation(self, user_input):
        """Run the conversation with the agent."""
        self.user_input = user_input

        # First, ensure we have a valid system message
        # if not any(isinstance(msg, SystemMessage) for msg in self.message):
        #     # Add default system message if none exists
        #     with open("prompts/main_prompt2.txt", "r", encoding="utf-8") as f:
        #         text = f.read()
        #         self.message.append(SystemMessage(content=user_input))

        # Add new user input
        self.message.append(HumanMessage(content=user_input))
        # self.message.append(("human",user_input))

        tools = create_tools(self)
        agent = create_agent(self.client, tools)
        agent_executor = create_agent_executor(agent, tools)
        # print("agent_executor", agent_executor)
        token_limit = 50000
        print("History: 2  ",self.message)
        def calculate_token_length(messages):
            num_tokens = 0
            for message in messages:
                num_tokens += 3
                num_tokens += len(self.encoding.encode(message.content))
            num_tokens += 3
            return num_tokens
        print("History: 3  ",self.message)
        def ensure_message_length_within_limit():
            conv_history_tokens = calculate_token_length(self.message)
            while conv_history_tokens + self.BEDROCK_MAX_TOKENS >= token_limit:
                if len(self.message) > 1:
                    # Keep the system message (first) and remove the second message
                    del self.message[1]
                    conv_history_tokens = calculate_token_length(self.message)
                else:
                    break

        # with get_openai_callback() as cb:
        #     print("get_openai_callback called")
        #     try:
        #         print("called 1")
        #         ensure_message_length_within_limit()
        #         print("called 2")
        #         print(
        #             agent_executor.invoke(
        #                 {"input": user_input, "chat_history": self.message}
        #             )
        #         )
        #         # async for chunk in agent_executor.astream_events(
        #         #     {"input": user_input, "chat_history": self.message}, version="v1"
        #         # ):
        #         #     # Process the chunk based on its type
        #         #     # print("called ", chunk)
        #         #     # if isinstance(chunk, dict):
        #         #     #     if "event" in chunk:
        #         #     #         event_type = chunk["event"]
        #         #     #         if event_type == "on_llm_stream":
        #         #     #             # Handle streaming LLM response
        #         #     #             if "data" in chunk and "chunk" in chunk["data"]:
        #         #     #                 response_chunk = chunk["data"]["chunk"]
        #         #     #                 # You can process the response chunk here
        #         #     #                 print(f"Response chunk: {response_chunk}")
        #         #     #         elif event_type == "on_tool_start":
        #         #     #             # Handle tool start event
        #         #     #             print(
        #         #     #                 f"Tool started: {chunk.get('data', {}).get('name')}"
        #         #     #             )
        #         #     #         elif event_type == "on_tool_end":
        #         #     #             # Handle tool end event
        #         #     #             print(
        #         #     #                 f"Tool ended: {chunk.get('data', {}).get('name')}"
        #         #     #             )
        #         #     #         elif event_type == "on_agent_action":
        #         #     #             # Handle agent action event
        #         #     #             print(f"Agent action: {chunk.get('data', {})}")
        #         #     time.sleep(0.05)
        #         #     yield chunk
        #     except Exception as e:
        #         error_message = f"An error occurred: {e}"
        #         print(error_message)
        #         yield {"error": error_message}

        with get_openai_callback() as cb:
            try:
 
                # ensure_message_length_within_limit()  # Reserve some tokens for functions and overhead
                # async for chunk in agent_executor.astream_events(
                #     {"input": user_input, "chat_history": self.message}, version="v1"
                # ):
                #     yield chunk
                print("History: 4  ",self.message)
                response = agent_executor.invoke({"input": user_input, "chat_history": self.message})
                print("History: =5  ",self.message)
                return response["output"]
 
            except Exception as e:
                error_message = f"An error occurred: {e}"
                print(error_message)
                return error_message

# ob1 = ABHFL()
# while True:
#     if len(ob1.message) == 0:
#         with open(r"prompts\\main_prompt1.txt", "r", encoding="utf-8") as f:
#             text = f.read()
#         ob1.message.append(("system",text))

#     question = input("Please Enter a Query: ")
#     if question == "end":
#         break

#     openapiresponse = ob1.run_conversation(question)
#     print(openapiresponse)

#     ob1.message.append(AIMessage(content=f"{openapiresponse}"))
#     # print(ob1.message)
