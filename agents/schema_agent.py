import os
import json
from typing import Dict, Any

from config.gemini_client import GeminiClient
from utils.json_parser import JSONParser


class SchemaAgent:
    """
    Schema Agent

    Responsibilities
    ----------------
    1. Load the schema prompt.
    2. Combine the prompt with dataset metadata.
    3. Send the request to Gemini.
    4. Parse Gemini's JSON response.
    5. Return a Python dictionary.
    """

    def __init__(self, prompt_path: str = "prompts/schema_prompt.txt") -> None:
        """
        Initialize the Schema Agent.

        Parameters
        ----------
        prompt_path : str
            Path to the schema prompt file.
        """

        self.prompt_path = prompt_path
        self.gemini_client = GeminiClient()

    def _load_prompt(self) -> str:
        """
        Load the schema prompt from file.

        Returns
        -------
        str
            Prompt text.
        """

        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(
                f"Prompt file not found: {self.prompt_path}"
            )

        with open(self.prompt_path, "r", encoding="utf-8") as file:
            return file.read()

    def analyze_schema(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze dataset metadata using Gemini.

        Parameters
        ----------
        dataset_info : Dict[str, Any]

        Returns
        -------
        Dict[str, Any]
            Parsed JSON response.
        """

        if not isinstance(dataset_info, dict):
            raise TypeError(
                "dataset_info must be a Python dictionary."
            )

        # Load system prompt
        prompt = self._load_prompt()

        # Convert Python dictionary into formatted JSON
        dataset_json = json.dumps(dataset_info, indent=4)

        # Build final prompt
        final_prompt = f"""
{prompt}

--------------------------------------------------

Dataset Metadata

{dataset_json}

--------------------------------------------------

Analyze the dataset metadata carefully.

Return ONLY valid JSON.
"""

        # Get Gemini response
        response = self.gemini_client.generate_response(final_prompt)

        # Convert JSON string into Python dictionary
        parsed_response = JSONParser.parse(response)

        return parsed_response