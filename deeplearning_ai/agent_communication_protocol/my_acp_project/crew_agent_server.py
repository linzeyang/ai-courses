"""crew_agent_server.py"""

from collections.abc import AsyncGenerator

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import RunYield, RunYieldResume, Server
from crewai import LLM, Agent, Crew, Task
from crewai_tools import RagTool

server = Server()

llm = LLM(
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/google/gemma-3-12b-it:free",
    timeout=60,
    temperature=0.7,
    max_tokens=1024,
    stream=False,
)

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "gemma3:1b",
            "temperature": 0.5,
            "base_url": "http://localhost:11434",
        },
    },
    "embedding_model": {
        "provider": "ollama",
        "config": {
            "model": "all-minilm:latest",
            "base_url": "http://localhost:11434",
        },
    },
}

rag_tool = RagTool(config=config, chunk_size=1200, chunk_overlap=200)  # type: ignore

rag_tool.add("../data/gold-hospital-and-premium-extras.pdf", data_type="pdf_file")


@server.agent()
async def policy_agent(
    _input: list[Message],
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    This is an agent for questions around policy coverage,
    it uses a RAG pattern to find answers based on policy documentation.
    Use it to help answer questions on coverage and waiting periods.
    """

    insurance_agent = Agent(
        role="Senior Insurance Coverage Assistant",
        goal="Determine whether something is covered or not",
        backstory="You are an expert insurance agent designed to assist with coverage queries",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[rag_tool],
        max_retry_limit=2,
    )

    task1 = Task(
        description=_input[0].parts[0].content or "dummy description",
        expected_output="A comprehensive response as to the users question",
        agent=insurance_agent,
    )

    crew = Crew(agents=[insurance_agent], tasks=[task1], verbose=True)

    task_output = await crew.kickoff_async()

    yield Message(role="agent", parts=[MessagePart(content=str(task_output))])


if __name__ == "__main__":
    server.run(port=8001)
