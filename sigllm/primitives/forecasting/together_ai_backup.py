# -*- coding: utf-8 -*-

import json
import os
import requests

from together import Together
from tqdm import tqdm

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TEMPLATE.json')

PROMPTS = json.load(open(PROMPT_PATH))


class TogetherAI:
    """Prompt Together AI models to forecast a time series.

    Args:
        name (str):
            Model name. Default to `'mistralai/Mistral-7B-Instruct-v0.2'`.
        sep (str):
            String to separate each element in values. Default to `','`.
        steps (int):
            Number of steps ahead to forecast. Default `1`.
        temp (float):
            Sampling temperature to use, between 0 and 2. Higher values like 0.8 will
            make the output more random, while lower values like 0.2 will make it
            more focused and deterministic. Default to `1`.
        top_p (float):
            Alternative to sampling with temperature, called nucleus sampling, where the
            model considers the results of the tokens with top_p probability mass.
            So 0.1 means only the tokens comprising the top 10% probability mass are
            considered. Default to `1`.
        samples (int):
            Number of forecasts to generate for each input message. Default to `1`.
        max_tokens (int):
            Maximum number of tokens to generate. Default to `50`.
    """

    def __init__(
        self,
        name='mistralai/Mistral-7B-Instruct-v0.2',
        sep=',',
        steps=1,
        temp=1,
        top_p=1,
        samples=1,
        max_tokens=50,
    ):
        self.name = name
        self.sep = sep
        self.steps = steps
        self.temp = temp
        self.top_p = top_p
        self.samples = samples
        self.max_tokens = max_tokens

        # Get API key from environment variable
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable is not set. "
                "Please set your Together AI API key: export TOGETHER_API_KEY='your-api-key'"
            )
        
        self.client = Together(api_key=api_key)

    def forecast(self, X, **kwargs):
        """Use Together AI to forecast a signal.

        Args:
            X (ndarray):
                Input sequences of strings containing signal values.

        Returns:
            list:
                List of forecasted signal values.
        """
        all_responses = []
        
        for text in tqdm(X):
            # Calculate appropriate max_tokens based on input
            input_length = len(text.split(','))
            calculated_max_tokens = max(self.max_tokens, input_length // 10 * self.steps)
            
            message = ' '.join([PROMPTS['user_message'], text, self.sep])
            
            responses = []
            for _ in range(self.samples):
                try:
                    response = self.client.chat.completions.create(
                        model=self.name,
                        messages=[
                            {'role': 'system', 'content': PROMPTS['system_message']},
                            {'role': 'user', 'content': message},
                        ],
                        max_tokens=calculated_max_tokens,
                        temperature=self.temp,
                        top_p=self.top_p,
                    )
                    content = response.choices[0].message.content
                    responses.append(content)
                    
                except Exception as e:
                    print(f"Error in Together AI API call: {e}")
                    responses.append("")
            
            all_responses.append(responses)

        return all_responses
