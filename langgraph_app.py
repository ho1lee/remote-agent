import asyncio
import os
from typing import TypedDict, List, Dict, Any, AsyncGenerator

import httpx
import gradio as gr
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langgraph.graph import StateGraph

# API Base URLs (replace with actual URLs if different)
LLM_BASE_URL = "http://ho1lee.aabb.cc"
RAG_BASE_URL = "http://ho2lee.aabb.cc"

# Example Payloads (adjust to actual API expectations)
LLM_PAYLOAD_TEMPLATE = {"model": "mistral-7b", "messages": [{"role": "user", "content": ""}]}
RAG_PAYLOAD_TEMPLATE = {"query": "", "top_k": 5}

# Example Headers (API keys will be added dynamically)
BASE_HEADERS = {"Content-Type": "application/json"}

# Define the state for our graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The user's input query.
        rag_response: The response from the RAG API.
        llm_response: The response from the LLM API.
        error_message: Optional error message if a step fails.
    """
    query: str
    rag_response: Dict[str, Any]
    llm_response: Dict[str, Any]
    error_message: str | None

# --- Main Application Logic will go here ---

def get_env_variable(var_name: str) -> str:
    """
    Retrieves an environment variable by its name.

    Args:
        var_name: The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not set.
    """
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not set. Please ensure it is available.")
    return value

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10), # Max wait 10s for faster feedback in this context
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    reraise=True # Reraise the exception if all retries fail
)
async def _make_api_call(client: httpx.AsyncClient, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to make an API call with httpx.AsyncClient."""
    response = await client.post(url, json=payload, headers=headers, timeout=10.0) # 10s timeout
    response.raise_for_status() # Raise HTTPStatusError for bad responses (4xx or 5xx)
    return response.json()

async def retrieve_rag(state: GraphState) -> GraphState:
    """
    Retrieves documents from the RAG API based on the query in the state.

    Args:
        state: The current graph state.

    Returns:
        The updated graph state with RAG response or an error message.
    """
    print("Node: retrieve_rag")
    query = state.get("query")
    if not query:
        return {**state, "error_message": "Query not found in state for RAG.", "rag_response": {}}

    try:
        rag_api_key = get_env_variable("RAG_API_KEY")
        headers = {**BASE_HEADERS, "Authorization": f"Bearer {rag_api_key}"}
        payload = {**RAG_PAYLOAD_TEMPLATE, "query": query}

        async with httpx.AsyncClient() as client:
            api_response = await _make_api_call(client, f"{RAG_BASE_URL}/query", headers, payload)

        return {**state, "rag_response": api_response, "error_message": None}
    except ValueError as e: # From get_env_variable
        print(f"RAG API Key error: {e}")
        return {**state, "error_message": str(e), "rag_response": {}}
    except httpx.HTTPStatusError as e:
        print(f"RAG API HTTP Status Error: {e.response.status_code} - {e.response.text}")
        return {**state, "error_message": f"RAG API Error: {e.response.status_code}", "rag_response": {}}
    except httpx.TimeoutException as e:
        print(f"RAG API Timeout Error: {e}")
        return {**state, "error_message": "RAG API request timed out.", "rag_response": {}}
    except Exception as e: # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred in retrieve_rag: {e}")
        return {**state, "error_message": f"Unexpected RAG error: {type(e).__name__}", "rag_response": {}}

async def call_llm(state: GraphState) -> GraphState:
    """
    Calls the LLM API with the query and context from RAG.

    Args:
        state: The current graph state.

    Returns:
        The updated graph state with LLM response or an error message.
    """
    print("Node: call_llm")
    query = state.get("query")
    rag_response = state.get("rag_response")

    if not query:
        return {**state, "error_message": "Query not found in state for LLM.", "llm_response": {}}

    try:
        llm_api_key = get_env_variable("LLM_API_KEY")
        headers = {**BASE_HEADERS, "Authorization": f"Bearer {llm_api_key}"}

        # Prepare context from RAG response
        context_str = ""
        if rag_response and isinstance(rag_response.get("documents"), list):
            # Assuming RAG returns a list of dicts with a "text" key or similar
            # This part might need adjustment based on actual RAG API response structure
            context_items = []
            for doc in rag_response["documents"][:3]: # Use top 3 documents
                if isinstance(doc, dict) and "text" in doc:
                    context_items.append(doc["text"])
                elif isinstance(doc, str): # If RAG returns list of strings
                    context_items.append(doc)
            if context_items:
                context_str = " ".join(context_items)
                context_str = "Context: " + context_str.strip() + "\n\n"


        # Prepare messages for LLM
        # Ensure LLM_PAYLOAD_TEMPLATE["messages"] is a list and has at least one item
        if not LLM_PAYLOAD_TEMPLATE.get("messages") or not isinstance(LLM_PAYLOAD_TEMPLATE["messages"], list):
             # Fallback if template is misconfigured
            messages = [{"role": "user", "content": f"{context_str}Question: {query}"}]
        else:
            # Use a copy of the message template to avoid modifying the global constant
            messages = [msg.copy() for msg in LLM_PAYLOAD_TEMPLATE["messages"]]
            # Update the content of the last message, assuming it's the user's message
            messages[-1]["content"] = f"{context_str}Question: {query}"

        payload = {**LLM_PAYLOAD_TEMPLATE, "messages": messages}

        async with httpx.AsyncClient() as client:
            api_response = await _make_api_call(client, f"{LLM_BASE_URL}/chat/completions", headers, payload)

        return {**state, "llm_response": api_response, "error_message": None}
    except ValueError as e: # From get_env_variable
        print(f"LLM API Key error: {e}")
        return {**state, "error_message": str(e), "llm_response": {}}
    except httpx.HTTPStatusError as e:
        print(f"LLM API HTTP Status Error: {e.response.status_code} - {e.response.text}")
        return {**state, "error_message": f"LLM API Error: {e.response.status_code}", "llm_response": {}}
    except httpx.TimeoutException as e:
        print(f"LLM API Timeout Error: {e}")
        return {**state, "error_message": "LLM API request timed out.", "llm_response": {}}
    except Exception as e:
        print(f"An unexpected error occurred in call_llm: {e}")
        return {**state, "error_message": f"Unexpected LLM error: {type(e).__name__}", "llm_response": {}}

async def run_graph_for_gradio(user_question: str) -> AsyncGenerator[str, None]:
    """
    Runs the LangGraph with the user's question and streams outputs for Gradio.
    """
    if not user_question or not user_question.strip():
        yield "Please enter a question."
        return

    initial_state = GraphState(
        query=user_question,
        rag_response={},
        llm_response={},
        error_message=None
    )

    yield "Starting process...\n---\n"
    current_output = ""

    try:
        async for event in APP_LANGGRAPH.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            node_name = event["name"] # Corresponds to node names given in add_node
            data = event.get("data", {})

            # Useful for debugging event stream:
            # print(f"Event: {event_type}, Node: {node_name}, Data Keys: {data.keys()}")

            if event_type == "on_chain_start":
                if node_name == "retrieve_rag":
                    current_output = "Retrieving documents...\n"
                    yield current_output
                elif node_name == "call_llm":
                    current_output += "Calling LLM...\n"
                    yield current_output

            elif event_type == "on_chain_end":
                # 'data' for on_chain_end contains 'output', which is the state *after* the node ran
                output_state = data.get("output")
                if not isinstance(output_state, dict): # Should be our GraphState dict
                    # This case should ideally not happen with proper graph setup
                    print(f"Warning: Output from node {node_name} is not a dict: {output_state}")
                    continue

                error_message = output_state.get("error_message")
                if error_message:
                    current_output += f"Error from {node_name}: {error_message}\n"
                    yield current_output
                    return # Stop generation on error

                if node_name == "retrieve_rag":
                    rag_resp = output_state.get("rag_response")
                    if rag_resp:
                        # You could add details from rag_resp if needed
                        current_output = "Documents retrieved successfully.\n" # Reset for next step
                        yield current_output
                    # If RAG failed, error_message above would have caught it.

                elif node_name == "call_llm":
                    llm_resp = output_state.get("llm_response")
                    if llm_resp:
                        try:
                            # Adjust this path based on the actual LLM API response structure
                            # Example: {"choices": [{"message": {"content": "..."}}]}
                            final_answer = llm_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                            if final_answer:
                                current_output += f"\nLLM Response:\n{final_answer}\n"
                            else:
                                current_output += "LLM response content not found in the expected structure.\n"
                            yield current_output
                        except (IndexError, AttributeError, TypeError) as e:
                            current_output += f"Error parsing LLM response: {e}. Response: {llm_resp}\n"
                            yield current_output
                            print(f"Problematic LLM response structure: {llm_resp}")
                    # If LLM failed, error_message would have caught it.

        current_output += "\n---\nProcess finished."
        yield current_output

    except Exception as e:
        # Catch-all for unexpected errors during graph streaming
        print(f"Error during graph execution for Gradio: {e}")
        yield f"\nAn unexpected error occurred: {e}"

# Construct the LangGraph
def build_graph() -> langgraph.graph.CompiledGraph:
    """Builds and compiles the LangGraph application."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("call_llm", call_llm)

    # Define edges
    graph.set_entry_point("retrieve_rag")
    graph.add_edge("retrieve_rag", "call_llm") # TODO: Consider conditional edge based on RAG success
    graph.set_finish_point("call_llm")

    # Compile the graph
    compiled_app = graph.compile()
    print("LangGraph compiled successfully.")
    return compiled_app

# Create the compiled app instance globally or pass it around
# For simplicity in Gradio, we can make it global or initialize in Gradio setup
# However, calling build_graph() each time Gradio runs a function is also an option,
# though less efficient if graph construction is heavy. Let's make it global for now.

APP_LANGGRAPH = build_graph()

def launch_gradio_interface():
    """
    Launches the Gradio web interface for interacting with the LangGraph app.
    """
    iface = gr.Interface(
        fn=run_graph_for_gradio,
        inputs=gr.Textbox(
            label="Your Question",
            placeholder="Type your question here and press Enter or click 'Submit'...",
            lines=4
        ),
        outputs=gr.Textbox(
            label="Answer Stream",
            lines=20,
            interactive=False,
            show_copy_button=True
        ),
        title="LangGraph RAG & LLM Application",
        description=(
            "Ask a question. The system will first retrieve relevant information (RAG) "
            "and then use a language model (LLM) to generate an answer based on your question and the retrieved context. "
            "API keys for RAG and LLM services must be set as environment variables (RAG_API_KEY, LLM_API_KEY)."
        ),
        allow_flagging="never", # Disable flagging
        # live=True # Consider if you want live updates as user types (might be too much for this app)
    )

    print("Launching Gradio interface...")
    # Use share=True if you want to create a public link (requires internet)
    # Use auth=("username", "password") for basic authentication
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    # Check for API keys before launching to provide a clear startup error if missing
    try:
        print("Checking for necessary API keys...")
        get_env_variable("LLM_API_KEY")
        get_env_variable("RAG_API_KEY")
        print("API keys found. Proceeding to launch Gradio interface.")

        # APP_LANGGRAPH is already built globally, so we can just launch
        launch_gradio_interface()

    except ValueError as e:
        print(f"Startup Error: {e}")
        print("Please set the required environment variables before running the application.")
        # Optionally, exit here if you don't want Gradio to launch at all
        # import sys
        # sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")
        # import sys
        # sys.exit(1)
