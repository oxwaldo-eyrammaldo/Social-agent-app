import os
from crewai import Agent, Task, Crew, Process

# Imports for LLM and Tools:
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# 1. Define the Tools
class SocialTools:
    @tool("Post Content")
    def post_content(content: str):
        """Mock posting content to social media."""
        return f"POST_SUCCESS: {content}"

@tool("Web Search Tool")
def search_web(query: str) -> str:
    """
    Search the web for up-to-date information on a given topic.
    The agent MUST use this tool to find trending news.
    """
    try:
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"[SEARCH ERROR] {str(e)}"

# 2. Crew Generator Function
def create_social_crew(topic, platform):

    # --- MOVED LLM SETUP INSIDE THE FUNCTION ---
    # This ensures it only runs AFTER the user enters their API key in Streamlit
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
        tools=[search_web],
        llm=openai_llm
    )

    manager = Agent(
        role='Social Media Manager',
        goal=f'Write a post for {platform}',
        backstory='You write punchy, engaging content.',
        verbose=True,
        allow_delegation=False,
        tools=[SocialTools.post_content],
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
