
import os
import uuid

from dotenv import load_dotenv
from google.adk.tools import ToolContext
from google.cloud import storage
from google.genai import Client, types

MODEL = "gemini-2.5-pro-preview-05-06" 
MODEL_IMAGE = "imagen-4.0-generate-preview-05-20"

load_dotenv()

# Only Vertex AI supports Imagen 4 for now.
client = Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

async def generate_image(img_prompt: str, tool_context: ToolContext):
    """Generates an image based on the prompt.
    
    Args:
        img_prompt (str): The prompt to generate the image from.
        tool_context (ToolContext): The tool context.

    Returns:
        dict: Status, detail and the filename of the image.

    """
    response = client.models.generate_images(
        model=MODEL_IMAGE,
        prompt=img_prompt,
        config={"number_of_images": 1},
    )
    if not response.generated_images:
        return {
            "status": "failed",
            "detail": "Image generation failed.",
        }
    image_bytes = response.generated_images[0].image.image_bytes
    filename = f"{uuid.uuid4()}.png"
    await tool_context.save_artifact(
        filename,
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
    )
    return {
        "status": "success",
        "detail": "Image generated successfully and stored in artifacts.",
        "filename": filename,
    }

async def upload_image_to_gcs(
    filename: str, tool_context: ToolContext, gcs_bucket: str = os.environ["BUCKET"]
):
    """
    Uploads an image to a GCS bucket.
    Args:
        filename (str): The name of the file to upload.
        tool_context (ToolContext): The tool context to use to load the artifact.
        gcs_bucket (str): The name of the GCS bucket to upload image to.
    Returns:
        str: The GCS URI of the uploaded image.
    """
    gcs_bucket = gcs_bucket.replace("gs://", "")
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)
    blob = bucket.blob(filename)
    # Get the image from ADK artifacts
    image_artifact = await tool_context.load_artifact(filename)
    if image_artifact and image_artifact.inline_data:
        print(f"Successfully loaded latest ADK artifact '{filename}'.")
        image_bytes = image_artifact.inline_data.data
        blob.upload_from_string(image_bytes, content_type="image/png")
    else:
        print(f"Failed to load ADK artifact '{filename}'.")
    return f"gs://{gcs_bucket}/{filename}"