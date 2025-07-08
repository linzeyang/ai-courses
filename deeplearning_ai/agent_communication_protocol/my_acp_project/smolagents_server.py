"""smolagents_server.py"""

from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server
from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents.agents import CodeAgent, ToolCallingAgent
from smolagents.default_tools import VisitWebpageTool, WebSearchTool
from smolagents.models import LiteLLMModel
from smolagents.tools import ToolCollection

load_dotenv()

server = Server()

model = LiteLLMModel(
    model_id="openrouter/openrouter/cypher-alpha:free",
    max_tokens=2048,
)

server_parameters = StdioServerParameters(
    command="python",
    args=["mcpserver.py"],
    env=None,
)


@server.agent()
async def health_agent(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    This is a CodeAgent which supports the hospital to handle health based questions for patients.
    Current or prospective patients can use it to find answers about their health and hospital treatments.
    """
    agent = CodeAgent(
        tools=[WebSearchTool(engine="bing"), VisitWebpageTool()], model=model
    )

    prompt = input[0].parts[0].content
    response = agent.run(task=prompt or "", stream=False)

    yield Message(role="agent", parts=[MessagePart(content=str(response))])


@server.agent()
async def doctor_agent(
    input: list[Message],
) -> AsyncGenerator[RunYield, RunYieldResume]:
    "This is a Doctor Agent which helps users find doctors near them."

    with ToolCollection.from_mcp(
        server_parameters, trust_remote_code=True
    ) as tool_collection:
        agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
        prompt = input[0].parts[0].content
        response = agent.run(task=prompt or "")

    yield Message(role="agent", parts=[MessagePart(content=str(response))])


if __name__ == "__main__":
    server.run(port=8000)
