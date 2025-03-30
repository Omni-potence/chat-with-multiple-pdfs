import os
from groq import Groq
from typing import List, Dict

class GroqAPI:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b"  # Using Llama 3 8B model
        self.conversation_history: List[Dict[str, str]] = []

    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, query: str, context: str = "") -> str:
        """Get a response from the Groq API using the conversation history and context."""
        # Prepare the messages for the API call
        messages = self.conversation_history.copy()
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Context: {context}\n\nPlease provide a response based on the above context and the user's query."
            })
        
        # Add the current query
        messages.append({"role": "user", "content": query})

        try:
            # Make the API call
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )

            # Extract the response
            response = completion.choices[0].message.content

            # Update conversation history
            self.add_to_history("user", query)
            self.add_to_history("assistant", response)

            return response
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return f"I encountered an error while generating a response. Please try again. Error: {str(e)}"

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []