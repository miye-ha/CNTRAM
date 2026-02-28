# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf

class LLMService_Deepseek:
    """
    OpenAI-style LLM service, responsible for interacting with DeepSeek API
    """
    def __init__(self, 
                 model_name: str = None, 
                 api_key: str = None, 
                 base_url: str = None):
        """
        Initialize OpenAI LLM service
        
        Args:
            model_name: Model name
            api_key: OpenAI API key
            base_url: OpenAI API base URL
        """
        self.model_name = model_name or conf.DEEPSEEK_MODEL_NAME
        self.client = OpenAI(
            api_key=api_key or conf.DEEPSEEK_API_KEY,
            base_url=base_url or conf.DEEPSEEK_BASE_URL,
        )
    
    def generate(self, 
                prompt: str, 
                system_prompt: str = None, 
                temperature: float = None,
                stream: bool = False
                ) -> Any:
        """
        Generate response
        
        Args:
            prompt: Prompt text
            system_prompt: System prompt
            temperature: Temperature parameter
            stream: Whether to stream output
            
        Returns:
            Any: Generated response
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt or conf.DEEPSEEK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or conf.DEEPSEEK_TEMPERATURE,
                stream=stream,
            )
            
            if stream:
                return self._handle_stream_response(response)
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Failed to generate response: {str(e)}")
            return None
    
    def _handle_stream_response(self, response: Any) -> Generator[str, None, None]:
        """
        Handle stream response
        
        Args:
            response: Stream response
            
        Returns:
            Generator: Generator
        """
        try:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Failed to handle stream response: {str(e)}")
            yield f"Error handling response: {str(e)}"
    
    def test_connection(self) -> bool:
        """
        Test connection
        
        Returns:
            bool: Whether connection is successful
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a AI assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                stream=False
            )
            return response
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False

if __name__ == "__main__":
    llm_service = LLMService_Deepseek()
    print(llm_service.generate("Hello, how are you?"))
