import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Create model
model = genai.GenerativeModel("gemini-2.5-flash")

# Send a test prompt
response = model.generate_content(
    "Reply with exactly this sentence: Gemini connection successful!"
)

# Print response
print("\nGemini Response:")
print(response.text)