'''This file contains the code that wraps the image search as a tool for an LLM agent'''
import os
import gradio as gr
import argparse
import logging
import time
from gradio import ChatMessage
import asyncio
from main import text_query
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

def image_search_tool(query: str) -> str:
    """
    This tool calls the image search functionality (text_query) and returns a textual summary
    of the results.
    """
    # Call the image search function with the provided query
    images, meta, map_fig = text_query(query)
    
    # Create a textual summary
    if not images:
        return "No images found for the query."
    
    summary = f"Found {len(images)} images.\n"
    summary += "Sample metadata (first few rows):\n"
    # Convert the first few rows of the metadata DataFrame to CSV (or plain text)
    summary += meta.head(3).to_csv(index=False)
    
    # (Optionally, you could save the map or gallery images and provide links.)
    return summary

# Create a LangChain Tool for image search
tools = [
    Tool(
        name="ImageSearch",
        func=image_search_tool,
        description=(
            "Use this tool to search for images based on a descriptive text query."
            "Input should be a plain text description of what images to find."
            "The tool uses a hybrid approach to find images based on a descriptive text query."
            "The hybrid approach incorporates vector search and a keyword (BM25F) search to find images."
            "This tool uses waeviate as the backend for vector storage and Image retrieval."
            "The keyword search in the tool will look in the fields called caption, camera, host, job, vsn, plugin, zone, project, and address."
            "The vector search in the tool will look in the fields called image and caption."
            "Weaviate will use Imagebind to generated vectors for images and texts."
        )
    )
]

# Initialize the LLM (OpenAI) with a temperature of 0 for deterministic output.
llm = OpenAI(temperature=0)
# Create the agent using a zero-shot chain that reacts to descriptions.
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

def llm_agent_interface(user_query: str) -> str:
    """
    This function receives a user query, lets the LLM agent decide how to use the tool,
    and returns the LLM’s response.
    """
    response = agent.run(user_query)
    return response

# Asynchronous generator function for interacting with the agent.
async def interact_with_image_search_agent(prompt, messages):
    # Append the user's message to the conversation.
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages

    # Call the synchronous llm_agent_interface in a thread to avoid blocking.
    response = await asyncio.to_thread(llm_agent_interface, prompt)
    messages.append(ChatMessage(role="assistant", content=response))
    yield messages
