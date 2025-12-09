import os
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun # Keep this for use in the wrapper function
from langchain_core.tools import tool

# 1. Define the Tools
class SocialTools:
    @tool("Post Content")
    def post_content(content: str):
        """Useful for posting the final content to social media."""
        # In a real app, this is where you'd call the Twitter/LinkedIn API
        return f"POST_SUCCESS: {content}"

# Define the search tool as a wrapped function to ensure compatibility
@tool("Web Search Tool")
def search_web(query: str) -> str:
    """
    Search the web for up-to-date information on a given topic.
    The agent MUST use this tool to find trending news.
    """
    # This function uses the DuckDuckGoSearchRun class to perform the actual search
    return DuckDuckGoSearchRun().run(query)

# Note: We no longer need 'search_tool = DuckDuckGoSearchRun()'

# 2. Define the Crew Generator Function
def create_social_crew(topic, platform):
    
    # -- Agents --
    researcher = Agent(
        role='Market Researcher',
        goal=f'Find trending news about {topic}',
        backstory='You are a researcher who finds viral news stories.',
        verbose=True,
        allow_delegation=False,
        tools=[search_web] # Pass the WRAPPED FUNCTION here
    )

    manager = Agent(
        role='Social Media Manager',
        goal=f'Write a post for {platform}',
        backstory='You write punchy, engaging content.',
        verbose=True,
        allow_delegation=False,
        tools=[SocialTools.post_content]
    )

    # ... (Tasks and Crew definition remain the same)
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
