import os

from dotenv import load_dotenv
from google import genai


class GeminiClient:
    """
    Gemini Client

    Responsible for:
    - Loading API key
    - Connecting to Gemini
    - Sending prompts
    - Returning AI responses
    """

    def __init__(self):

        # Load environment variables
        load_dotenv()

        # Read API key
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in .env file.")

        # Create Gemini client
        self.client = genai.Client(api_key=api_key)

        # Model name
        self.model = "gemini-2.5-flash"

    def generate_response(self, prompt):
        """
        Send a prompt to Gemini and return the response.
        """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text