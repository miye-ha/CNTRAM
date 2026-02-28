# -*- coding: utf-8 -*-
from ollama import Client
from typing import Dict, Any, Optional, Generator
import openai
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import conf

class Ccc_code(BaseModel):
    codes: list[Optional[str]]


class LLMService_Ollama:
    """
    LLM service, responsible for interacting with large language models
    """
    def __init__(self, base_url: str = None, keep_alive: str = None):
        """
        Initialize LLM service
        
        Args:
            base_url: Ollama service address
            keep_alive: Keep connection time
        """
        self.base_url = base_url or conf.OLLAMA_BASE_URL
        self.keep_alive = keep_alive or conf.OLLAMA_KEEP_ALIVE
        self.client = Client(
            host=self.base_url,
            headers={'x-some-header':'some-value'},
        )
    
    def generate(self, 
                prompt: str, 
                model_name: str = None,
                system_prompt: str = None, 
                temperature: float = None,
                stream: bool = False,
                format_schema: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generate response
        
        Args:
            prompt: Prompt text
            model_name: Model name
            system_prompt: System prompt
            temperature: Temperature parameter
            stream: Whether to stream output
            format_schema: Output format schema
            
        Returns:
            Any: Generated response
        """
        try:
            response = self.client.chat(
                model=model_name or conf.OLLAMA_MODEL_NAME,
                options={
                    "temperature": temperature or conf.OLLAMA_TEMPERATURE,
                    "seed": conf.OLLAMA_SEED,
                    "repeat_penalty": conf.OLLAMA_REPEAT_PENALTY,
                },
                format=Ccc_code.model_json_schema(),
                messages=[{
                    "role": "system",
                    "content": system_prompt or conf.OLLAMA_SYSTEM_PROMPT
                },{
                    "role": "user",
                    "content": prompt,
                }],
                stream=stream,
                keep_alive=self.keep_alive,
            )
            
            if stream:
                return self._handle_stream_response(response)
            content = response.message.content
            content = content.split('```')[-1]
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
                if 'message' in chunk:
                    yield chunk['message']['content']
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
            response = self.client.chat(
                model=conf.OLLAMA_MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": "You are a AI assistant."
                }, {
                    "role": "user",
                    "content": "Hello!"
                }],
                stream=False
            )
            return True
        except Exception as e:
            print(f"Connection test failed: {str(e)}")
            return False

if __name__ == "__main__":
    llm_service = LLMService_Ollama()
    print(llm_service.generate("Hello, how are you?"))
