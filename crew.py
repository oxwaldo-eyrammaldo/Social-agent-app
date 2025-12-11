import os
from crewai import Agent, Task, Crew, Process

# Imports for LLM and Tools:
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool  # We use the Tool class instead of the decorator

# 1. Define Tool Functions (No Decorators)
def post_content_func(content: str):
    """Mock posting content to social media."""
    return f"POST_SUCCESS: {content}"

def search_web_func(query: str):
    """
    Search the web for up-to-date information on a given topic.
    The agent MUST use this tool to find trending news.
    """
    try:
        search_tool = DuckDuckGoSearchRun()
        return search_tool.run(query)
    except Exception as e:
        return f"[SEARCH ERROR] {str(e)}"

# 2. Wrap them in the Tool Class
# This explicitly creates the tool object CrewAI expects
post_tool = Tool(
    name="Post Content",
    func=post_content_func,
    description="Mock posting content to social media."
)

search_tool = Tool(
    name="Web Search Tool",
    func=search_web_func,
    description="Search the web for up-to-date information. Useful for finding trending news."
)


# 3. Crew Generator Function
def create_social_crew(topic, platform):

    # -- LLM Setup --
    # Defined here so it runs AFTER the API key is set in Streamlit
    openai_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # -- Agents --
    researcher = Agent(
        role='Market Researcher',
        goal=f'Find trending news about {topic}',
        backstory='You are a researcher who finds viral news stories.',
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],  # Use the manually created tool
        llm=openai_llm
    )

    manager = Agent(
        role='Social Media Manager',
        goal=f'Write a post for {platform}',
        backstory='You write punchy, engaging content.',
        verbose=True,
        allow_delegation=False,
        tools=[post_tool],    # Use the manually created tool
        llm=openai_llm
    )

    # -- Tasks --
    task_research = Task(
        description=f'Find 2 interesting recent news items about {topic}.',
        expected_output='A summary of the news.',
        agent=researcher
    )

    task_write = Task(
        description=f'Write a {platform} post based on the research. Keep it under 280 chars if Twitter.',
        expected_output='The final post text.',
        agent=manager
    )

    # -- Crew --
    crew = Crew(
        agents=[researcher, manager],
        tasks=[task_research, task_write],
        process=Process.sequential
    )

    return crew
