"""ADK Agent for creating images using Imagen 4 and Videos using Veo 2"""

from google.adk import Agent
from google.adk.tools import load_artifacts
from .tools import generate_image, upload_image_to_gcs

MODEL = "gemini-2.5-pro-preview-05-06" 

root_agent = Agent(
    model=MODEL,
    name="content_agent",
    instruction="You are an agent whose job is to generate or edit an image based on the prompt provided",
    tools=[generate_image, upload_image_to_gcs, load_artifacts],
)