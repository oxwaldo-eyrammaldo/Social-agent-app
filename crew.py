# crew.py
import os
from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun


# ------------------------------------------------------
# 1) LLM SETUP
# ------------------------------------------------------

def get_llm():
    """
    Safe LLM initializer that prevents deployment failures.
    """
    try:
        return ChatOpenAI(
            model="gpt-4o-mini",  # cheap & compatible
            temperature=0.3,
            max_tokens=3000,
        )
    except Exception as e:
        # Fail-safe mock response if OpenAI key is not set
        class MockLLM:
            def invoke(self, prompt):
                return f"[MOCK LLM RESPONSE due to error: {str(e)}]\nPrompt: {prompt}"
        return MockLLM()


llm = get_llm()


# ------------------------------------------------------
# 2) SEARCH TOOL (with safe fallback)
# ------------------------------------------------------

def load_search_tool():
    """
    Returns DuckDuckGo tool if available, else a safe mock.
    """
    try:
        return DuckDuckGoSearchRun()
    except Exception as e:
        class MockSearch:
            def run(self, query):
                return f"[MOCK SEARCH] No search available.\nQuery: {query}"
        return MockSearch()


search_tool = load_search_tool()


# ------------------------------------------------------
# 3) AGENT DEFINITIONS
# ------------------------------------------------------

research_agent = Agent(
    name="Research Analyst",
    role="Investigates and gathers latest online information.",
    goal="Provide accurate, concise research from reliable sources.",
    backstory=(
        "You are a detail-oriented research expert who finds factual, "
        "traceable online information and summarizes it clearly."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True,
)


writer_agent = Agent(
    name="Content Writer",
    role="Transforms research into high-quality narratives.",
    goal="Convert research findings into engaging, well-structured content.",
    backstory=(
        "You are an exceptional writer skilled in summarizing, rewriting, "
        "and generating content in a professional tone."
    ),
    llm=llm,
    verbose=True,
)


# ------------------------------------------------------
# 4) TASKS
# ------------------------------------------------------

research_task = Task(
    description=(
        "Search the web and compile key findings for the following topic:\n"
        "{topic}\n\n"
        "Return bullet points containing facts, stats, trends, and insights."
    ),
    expected_output="A structured research summary in bullet-point format.",
    tools=[search_tool],
    agent=research_agent,
    context={}
)


writing_task = Task(
    description=(
        "Using the research summary, write a clean, engaging narrative. "
        "Keep it professional, factual, and easy to read."
    ),
    expected_output="A polished final article (4–8 paragraphs).",
    agent=writer_agent,
)


# ------------------------------------------------------
# 5) CREW PIPELINE
# ------------------------------------------------------

class ResearchCrew:
    def __init__(self, topic: str):
        self.topic = topic

        # inject context
        research_task.context = {"topic": topic}

        # build crew
        self.crew = Crew(
            agents=[research_agent, writer_agent],
            tasks=[research_task, writing_task],
            verbose=True,
        )

    def run(self):
        """
        Executes both tasks sequentially.
        """
        try:
            return self.crew.run()
        except Exception as e:
            return f"[CREW ERROR] {str(e)}"
