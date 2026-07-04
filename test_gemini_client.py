from config.gemini_client import GeminiClient


client = GeminiClient()

response = client.generate_response(
    "Reply with exactly this sentence: Gemini Client Working Successfully!"
)

print("\nGemini Response:\n")
print(response)