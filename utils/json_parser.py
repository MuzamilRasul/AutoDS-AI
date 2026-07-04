import json
import re


class JSONParser:
    """
    JSON Parser Utility

    Responsible for:
    - Cleaning Gemini responses
    - Removing Markdown code blocks
    - Converting JSON text into a Python dictionary
    """

    @staticmethod
    def parse(response_text):
        """
        Convert Gemini response into a Python dictionary.

        Parameters:
            response_text (str)

        Returns:
            dict
        """

        try:
            # Remove ```json
            cleaned_text = re.sub(r"```json", "", response_text)

            # Remove ```
            cleaned_text = re.sub(r"```", "", cleaned_text)

            # Remove leading/trailing spaces
            cleaned_text = cleaned_text.strip()

            # Convert JSON string to Python dictionary
            return json.loads(cleaned_text)

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON returned by Gemini:\n\n{response_text}"
            ) from e