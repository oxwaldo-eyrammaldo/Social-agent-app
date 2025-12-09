import os
from crewai import Agent, Task, Crew, Process

# Imports for LLM and Tools:
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool # Corrected decorator import

# 0. Define the LLM Instance (using a model you have access to)
openai_llm = ChatOpenAI(model='gpt-3.5-turbo') 

# 1. Define the Tools
class SocialTools:
    @tool("Post Content")
    def post_content(content: str):
        """Useful for posting the final content to social media."""
        # In a real app, this is where you'd call the Twitter/LinkedIn API
        return f"POST_SUCCESS: {content}"

# Define the search tool as a wrapped function for compatibility and validation
@tool("Web Search Tool")
def search_web(query: str) -> str:
    """
    Search the web for up-to-date information on a given topic.
    The agent MUST use this tool to find trending news.
    """
    # This uses the DuckDuckGoSearchRun class to perform the actual search
    return DuckDuckGoSearchRun().run(query)


# 2. Define the Crew Generator Function
def create_social_crew(topic, platform):
    
    # -- Agents --
    researcher = Agent(
        role='Market Researcher',
        goal=f'Find trending news about {topic}',
        backstory='You are a researcher who finds viral news stories.',
        verbose=True,
        allow_delegation=False,
        tools=[search_web], # Use the wrapped search function
        llm=openai_llm      # Assign the accessible LLM
    )

    manager = Agent(
        role='Social Media Manager',
        goal=f'Write a post for {platform}',
        backstory='You write punchy, engaging content.',
        verbose=True,
        allow_delegation=False,
        tools=[SocialTools.post_content],
        llm=openai_llm      # Assign the accessible LLM
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
