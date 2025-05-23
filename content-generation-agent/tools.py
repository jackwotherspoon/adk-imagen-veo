
import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
from google.adk.tools import ToolContext
from google.cloud import storage
from google.genai import Client, types

MODEL_IMAGE = "imagen-4.0-generate-preview-05-20"
MODEL_VIDEO = "veo-2.0-generate-preview"

load_dotenv()

logger = logging.getLogger(__name__)

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

async def generate_video(
    video_prompt: str,
    tool_context: ToolContext,
    number_of_videos: int = 1,
    aspect_ratio: str = "16:9",
    negative_prompt: str = "",
    existing_image_filename: str = "",
):
    """Generates a video based on the prompt.

    Args:
        video_prompt (str): The prompt to generate the video from.
        tool_context (ToolContext): The tool context.
        number_of_videos (int, optional): The number of videos to generate. Defaults to 1.
        aspect_ratio (str, optional): The aspect ratio of the video. Defaults to "16:9".
        negative_prompt (str, optional): The negative prompt to use. Defaults to "".

    Returns:
        dict: status dict

    Supported aspect ratios are:
        16:9 (landscape) and 9:16 (portrait) are supported.
    """
    gen_config = types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        number_of_videos=number_of_videos,
        output_gcs_uri=os.environ["BUCKET"],
        negative_prompt=negative_prompt,
    )
    # If an existing image is provided, use it to generate the video.
    if existing_image_filename:
        gcs_location = f"{os.environ['BUCKET']}/{existing_image_filename}"
        existing_image = types.Image(gcs_uri=gcs_location, mime_type="image/png")
        operation = client.models.generate_videos(
            model=MODEL_VIDEO,
            prompt=video_prompt,
            image=existing_image,
            config=gen_config,
        )
    else:
        operation = client.models.generate_videos(
            model=MODEL_VIDEO, prompt=video_prompt, config=gen_config
        )
    while not operation.done:
        logger.info("--- ⏳ Waiting for video generation to be done... ---")
        await asyncio.sleep(10)
        operation = client.operations.get(operation)
        print(operation)

    if operation.response:
        # Download and save the generated videos to artifacts.
        for generated_video in operation.result.generated_videos:
            video_uri = generated_video.video.uri
            filename = f"{uuid.uuid4()}.mp4"
            bucket = os.getenv("BUCKET")
            video_bytes = download_blob_from_gcs(
                bucket.replace("gs://", ""),
                video_uri.replace(bucket, "")[1:],  # get rid of trailing slash
            )
            print(f" --- 🗄️ The location for the saved video is here: {filename} --- ")
            await tool_context.save_artifact(
                filename,
                types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
            )
        return {
            "status": "success",
            "detail": "Video generated successfully and stored in artifacts.",
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
        logger.info(f" --- ✅ Successfully loaded latest ADK artifact '{filename}' --- ")
        image_bytes = image_artifact.inline_data.data
        blob.upload_from_string(image_bytes, content_type="image/png")
    else:
        logger.info(f" --- ❌ Failed to load ADK artifact '{filename}'. --- ")
    return f"gs://{gcs_bucket}/{filename}"

def download_blob_from_gcs(bucket_name: str, source_blob_name: str):
    """Downloads a blob from the Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()