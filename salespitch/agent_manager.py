from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from .prompts import sales_pitch_prompt

MEMORY_KEY = "chat_history"

def create_agent(llm, tools):
    """Create an agent with the given language model and tools."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert conversational sales manager with access to various tools to deliver clear and concise answers. 
                Select the most relevant tool for a precise response, avoiding unnecessary details. 
                When responding to general or open-ended questions, always leverage tools for accuracy. 
                If unsure of an answer, ask follow-up questions to clarify. You are experienced and professional in this role.""",
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return agent


def create_agent_executor(agent, tools, verbose=True):
    """Create an agent executor with the given agent and tools."""
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    return agent_executor

def create_sales_pitch_agent(llm):
    """Create a sales pitch agent."""
    sales_pitch = sales_pitch_prompt()
    if sales_pitch:
        prompt = ChatPromptTemplate.from_messages([
            ("system", sales_pitch),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        # No tools needed for sales pitch agent
        agent = create_tool_calling_agent(llm, [], prompt)
        return agent
    return None
