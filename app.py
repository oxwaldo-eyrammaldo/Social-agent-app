import streamlit as st
from crew import create_social_crew
import os

# Page Config
st.set_page_config(page_title="AI Social Manager", page_icon="🤖")

st.title("🤖 Agentic Social Media Manager")
st.markdown("Enter a topic, and your AI crew will research and write a post for you.")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# Inputs
topic = st.text_input("What topic should we post about?", "Generative AI")
platform = st.selectbox("Select Platform", ["Twitter/X", "LinkedIn", "Instagram"])

# The "Run" Button
if st.button("Generate & Post"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        with st.spinner('🤖 The Agents are working... (Researching & Writing)'):
            try:
                # Initialize the Crew
                social_crew = create_social_crew(topic, platform)
                
                # Run the Agents
                result = social_crew.kickoff()
                
                # Display Results
                st.success("Task Complete!")
                st.subheader("📝 Generated Post:")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")