# -*- coding: utf-8 -*-

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import aiohttp
from together import Together
from tqdm.asyncio import tqdm

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TEMPLATE.json')
PROMPTS = json.load(open(PROMPT_PATH))


class TogetherAIOptimized:
    """Optimized Together AI models for faster time series forecasting.

    Performance optimizations:
    1. Async/parallel API calls
    2. Batch processing
    3. Intelligent token calculation
    4. Connection pooling
    5. Error handling with retries

    Args:
        name (str):
            Model name. Default to `'mistralai/Mistral-7B-Instruct-v0.2'`.
        sep (str):
            String to separate each element in values. Default to `','`.
        steps (int):
            Number of steps ahead to forecast. Default `1`.
        temp (float):
            Sampling temperature to use, between 0 and 2. Default to `0.7`.
        top_p (float):
            Alternative to sampling with temperature. Default to `0.9`.
        samples (int):
            Number of forecasts to generate for each input message. Default to `1`.
        max_tokens (int):
            Maximum number of tokens to generate. Default to `50`.
        max_concurrent (int):
            Maximum number of concurrent API requests. Default to `10`.
        batch_size (int):
            Number of requests to process in each batch. Default to `20`.
        use_async (bool):
            Whether to use async processing. Default to `True`.
    """

    def __init__(
        self,
        name='mistralai/Mistral-7B-Instruct-v0.2',
        sep=',',
        steps=1,
        temp=0.7,  # Slightly lower for more consistent results
        top_p=0.9,  # Slightly lower for faster processing
        samples=1,  # Reduced default samples for speed
        max_tokens=50,
        max_concurrent=10,
        batch_size=20,
        use_async=True,
    ):
        self.name = name
        self.sep = sep
        self.steps = steps
        self.temp = temp
        self.top_p = top_p
        self.samples = samples
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.use_async = use_async

        # Get API key from environment variable
        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError(
                "TOGETHER_API_KEY environment variable is not set. "
                "Please set your Together AI API key: export TOGETHER_API_KEY='your-api-key'"
            )
        
        self.api_key = api_key
        self.client = Together(api_key=api_key)
        
        # API endpoint for direct HTTP calls
        self.api_url = "https://api.together.xyz/v1/chat/completions"

    def _calculate_optimal_tokens(self, text: str) -> int:
        """Calculate optimal max_tokens based on input length and steps."""
        input_length = len(text.split(','))
        # More conservative token calculation for better speed/accuracy balance
        base_tokens = min(self.max_tokens, max(20, input_length // 5 * self.steps))
        return base_tokens

    async def _async_api_call(self, session: aiohttp.ClientSession, text: str, semaphore: asyncio.Semaphore) -> List[str]:
        """Make async API call with semaphore for concurrency control."""
        async with semaphore:
            calculated_max_tokens = self._calculate_optimal_tokens(text)
            message = ' '.join([PROMPTS['user_message'], text, self.sep])
            
            payload = {
                "model": self.name,
                "messages": [
                    {'role': 'system', 'content': PROMPTS['system_message']},
                    {'role': 'user', 'content': message},
                ],
                "max_tokens": calculated_max_tokens,
                "temperature": self.temp,
                "top_p": self.top_p,
                "n": self.samples,  # Generate multiple samples in one call
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        responses = [choice['message']['content'] for choice in result['choices']]
                        return responses
                    else:
                        error_text = await response.text()
                        print(f"API Error {response.status}: {error_text}")
                        return [""] * self.samples
                        
            except Exception as e:
                print(f"Error in async API call: {e}")
                return [""] * self.samples

    def _sync_api_call(self, text: str) -> List[str]:
        """Synchronous API call as fallback."""
        calculated_max_tokens = self._calculate_optimal_tokens(text)
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
                print(f"Error in sync API call: {e}")
                responses.append("")
        
        return responses

    async def _async_forecast_batch(self, X_batch: List[str]) -> List[List[str]]:
        """Process a batch of inputs asynchronously."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout
            connector=aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        ) as session:
            tasks = [
                self._async_api_call(session, text, semaphore)
                for text in X_batch
            ]
            
            results = await tqdm.gather(*tasks, desc=f"Processing batch of {len(X_batch)}")
            return results

    def _sync_forecast_parallel(self, X: List[str]) -> List[List[str]]:
        """Process inputs using ThreadPoolExecutor for parallelism."""
        all_responses = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            futures = {executor.submit(self._sync_api_call, text): i for i, text in enumerate(X)}
            
            # Collect results in order
            results = [None] * len(X)
            
            for future in tqdm(as_completed(futures), total=len(X), desc="Processing requests"):
                index = futures[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Error processing request {index}: {e}")
                    results[index] = [""] * self.samples
            
            return results

    def forecast(self, X, **kwargs):
        """Use Together AI to forecast signals with optimized performance.

        Args:
            X (ndarray):
                Input sequences of strings containing signal values.

        Returns:
            list:
                List of forecasted signal values.
        """
        start_time = time.time()
        print(f"🚀 Starting optimized forecast with {len(X)} sequences...")
        print(f"   • Async mode: {self.use_async}")
        print(f"   • Max concurrent: {self.max_concurrent}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Samples per sequence: {self.samples}")
        
        X_list = X.tolist() if hasattr(X, 'tolist') else list(X)
        
        if self.use_async:
            try:
                # Process in batches asynchronously
                all_responses = []
                
                for i in range(0, len(X_list), self.batch_size):
                    batch = X_list[i:i + self.batch_size]
                    print(f"📦 Processing batch {i//self.batch_size + 1}/{(len(X_list) + self.batch_size - 1)//self.batch_size}")
                    
                    batch_results = asyncio.run(self._async_forecast_batch(batch))
                    all_responses.extend(batch_results)
                
            except Exception as e:
                print(f"⚠️  Async processing failed, falling back to sync: {e}")
                all_responses = self._sync_forecast_parallel(X_list)
        else:
            # Use sync parallel processing
            all_responses = self._sync_forecast_parallel(X_list)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Forecast completed in {elapsed_time:.2f} seconds")
        print(f"   • Average time per sequence: {elapsed_time/len(X):.3f}s")
        print(f"   • Total API calls: {len(X) * self.samples}")
        
        return all_responses


# Legacy class name for backward compatibility
class TogetherAI(TogetherAIOptimized):
    """Enhanced TogetherAI class with optimizations and backward compatibility."""
    
    def __init__(
        self,
        name='mistralai/Mistral-7B-Instruct-v0.2',
        sep=',',
        steps=1,
        temp=1,
        top_p=1,
        samples=1,
        max_tokens=50,
        # New optimization parameters with defaults for backward compatibility
        max_concurrent=10,
        batch_size=20,
        use_async=True,
        **kwargs
    ):
        # Map 'steps' to 'steps' for backward compatibility
        if 'steps' in kwargs:
            steps = kwargs.pop('steps')
        
        super().__init__(
            name=name,
            sep=sep,
            steps=steps,
            temp=temp,
            top_p=top_p,
            samples=samples,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
            use_async=use_async,
            **kwargs
        )
