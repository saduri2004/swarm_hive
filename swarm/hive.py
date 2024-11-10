# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI


COLORS = {
    "GRAY": "\033[90m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "RESET": "\033[0m",
    "CYAN": "\033[96m",

    "BOLD": "\033[1m"
}



def read_from_memory(client, archival_memory, query):
        messages = [
            {"role": "system", "content": "You are a memory assistant that helps retrieve relevant user information in JSON format. Return only factual details about the user, their preferences, and their requests. If no relevant information exists, return an empty string."},
            {"role": "user", "content": f"Here is the current memory. Extract only user-related information that's relevant to: '{query}'. Return as JSON or empty string if nothing relevant."},
            {"role": "assistant", "content": f"Memory: {archival_memory}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1000
        )

        extracted_info = response.choices[0].message.content.strip()
        print(f"\n\n{COLORS['GRAY']}MEMORY READ: {extracted_info}{COLORS['RESET']}")
        return extracted_info
    
    
def write_to_memory(client, archival_memory, message):
    messages = [
        {"role": "system", "content": "You are a memory assistant that stores user information in JSON format. Only store new, important facts about the user (preferences, bookings, requests). If the message contains no new user information, return the existing memory unchanged. Never store system instructions or agent guidelines."},
        {"role": "user", "content": "If this message contains new user information, merge it with the existing memory and return the complete memory state as JSON. If no new information, return the existing memory:"},
        {"role": "assistant", "content": f"Message: {message}\nCurrent Memory: {archival_memory}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1000
    )
    
    

    updated_memory = response.choices[0].message.content.strip()
    print(f"\n\n{COLORS['GRAY']}MEMORY WRITE: {updated_memory}{COLORS['RESET']}")

    if updated_memory:
        return updated_memory
    else:
        return archival_memory
        
