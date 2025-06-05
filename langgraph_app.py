import asyncio
import os
from typing import TypedDict, List, Dict, Any, AsyncGenerator

import json # Add this to the imports at the top
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

        # Fields for "Create FA Report" flow
        fail_name: str | None
        fa_report_name: str | None

        # Field for displaying the full state in the UI # Removed from active fields
        # serialized_graph_state: str | None # This line is now fully removed from active GraphState

        # Fields for Document Browser
        retrieved_docs_for_browser: List of documents with title and content.
        selected_doc_content: Content of the currently selected document.
    """
    query: str | None # Query can be None if fail_name is used
    rag_response: Dict[str, Any] # Raw RAG response
    llm_response: Dict[str, Any] # Raw LLM response
    error_message: str | None

    # Fields for "Create FA Report" flow
    fail_name: str | None
    fa_report_name: str | None

    # Fields for Document Browser in "Search Docs" tab
    retrieved_docs_for_browser: List[Dict[str, str]] | None
    selected_doc_content: str | None

    # serialized_graph_state: str | None # Ensure this is not an active field

# --- Main Application Logic will go here ---

# Placeholder for state serialization helper (will be refined in a later step)
def _serialize_state_to_str(state: GraphState) -> str:
    """Serializes the graph state to a pretty-printed JSON string."""
    # Filter out potentially large or complex fields if necessary for display
    # Also explicitly exclude 'serialized_graph_state' in case older dicts are passed.
    display_state = {k: v for k, v in state.items() if k not in ["rag_response", "llm_response", "serialized_graph_state"]}
    # Add summaries or key info from rag_response and llm_response if needed
    if state.get("rag_response") and isinstance(state.get("rag_response"), dict):
        display_state["rag_summary"] = {
            "document_count": len(state["rag_response"].get("documents", [])),
            "keys": list(state["rag_response"].keys())
        }
    if state.get("llm_response") and isinstance(state.get("llm_response"), dict):
        # Avoid serializing full LLM potentially large response directly
        display_state["llm_summary"] = {
            "keys": list(state["llm_response"].keys()),
            "choice_1_content_preview": state["llm_response"].get("choices", [{}])[0].get("message",{}).get("content","")[:100]+"..."
        }

    return json.dumps(display_state, indent=2, default=str) # default=str for any non-serializable items

async def run_search_docs_flow(user_query: str) -> AsyncGenerator[tuple[List[str], str, str, List[Dict[str, str]] | None], None]:
    """
    Runs the 'Search Docs' flow (RAG only) for the document browser.
    Yields: (titles_list, selected_doc_content, serialized_graph_state, all_retrieved_docs)
    """
    # Initial state setup (ensure new GraphState fields are defaulted)
    initial_state = GraphState(
        query=user_query,
        rag_response={},
        llm_response={},
        error_message=None,
        fail_name=None,
        fa_report_name=None,
        retrieved_docs_for_browser=None,
        selected_doc_content=None
    )

    if not user_query or not user_query.strip():
        no_query_msg = "Please enter a query to search."
        empty_docs_list: List[Dict[str,str]] = []
        # Ensure initial_state here also has the error if needed by _serialize_state_to_str
        initial_state["error_message"] = no_query_msg
        yield ([], no_query_msg, _serialize_state_to_str(initial_state), empty_docs_list)
        return

    # Initial yield for "Processing..." message
    yield (["Searching..."], "Searching...", _serialize_state_to_str(initial_state), None)

    processed_docs_for_browser: List[Dict[str, str]] = []
    titles_for_radio: List[str] = []
    current_selected_content: str = "No documents found yet."
    # Cast initial_state to GraphState for type safety if it's being passed around and modified.
    # However, final_output_state will be reassigned to the output of the graph.
    final_output_state: GraphState = initial_state

    try:
        async for event in SEARCH_DOCS_GRAPH.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            node_name = event["name"]
            data = event.get("data", {})

            if event_type == "on_chain_start":
                if node_name == "retrieve_rag":
                    yield (["Retrieving documents..."], "Please wait...", _serialize_state_to_str(initial_state), None)

            elif event_type == "on_chain_end":
                output_state_from_event = data.get("output")
                if not isinstance(output_state_from_event, dict): continue

                # Ensure all keys are present, effectively casting to GraphState
                final_output_state = GraphState(
                    query=output_state_from_event.get("query", initial_state["query"]),
                    rag_response=output_state_from_event.get("rag_response", initial_state["rag_response"]),
                    llm_response=output_state_from_event.get("llm_response", initial_state["llm_response"]),
                    error_message=output_state_from_event.get("error_message", initial_state["error_message"]),
                    fail_name=output_state_from_event.get("fail_name", initial_state["fail_name"]),
                    fa_report_name=output_state_from_event.get("fa_report_name", initial_state["fa_report_name"]),
                    retrieved_docs_for_browser=output_state_from_event.get("retrieved_docs_for_browser"), # Might be None initially
                    selected_doc_content=output_state_from_event.get("selected_doc_content") # Might be None initially
                )

                error_message_from_state = final_output_state.get("error_message")
                if error_message_from_state:
                    current_selected_content = f"Error: {error_message_from_state}"
                    # final_output_state already contains the error.
                    break # Break from loop to yield error state outside

                if node_name == "retrieve_rag":
                    raw_docs = final_output_state.get("rag_response", {}).get("documents", [])
                    processed_docs_for_browser = []
                    for i, doc_data in enumerate(raw_docs[:10]): # Max 10 hits
                        if not isinstance(doc_data, dict): continue

                        title = doc_data.get("title", doc_data.get("name"))
                        content = doc_data.get("content", doc_data.get("text", "Content not available."))
                        if not title:
                            title = content[:50] + "..." if content else f"Document {i+1}"

                        processed_docs_for_browser.append({"title": title, "content": content})

                    final_output_state["retrieved_docs_for_browser"] = processed_docs_for_browser

                    if processed_docs_for_browser:
                        titles_for_radio = [doc["title"] for doc in processed_docs_for_browser]
                        current_selected_content = processed_docs_for_browser[0]["content"]
                        final_output_state["selected_doc_content"] = current_selected_content
                    else:
                        titles_for_radio = []
                        current_selected_content = "No documents found for the query."
                        final_output_state["selected_doc_content"] = current_selected_content
                    # This yield is inside the loop, specifically after RAG.
                    # It updates the UI as soon as documents are processed.
                    yield (titles_for_radio, current_selected_content, _serialize_state_to_str(final_output_state), processed_docs_for_browser)

        # After loop processing (either completed or broke due to error)
        if final_output_state.get("error_message"):
             # Error message is already in current_selected_content if loop broke from error path
             # If error happened outside loop or was set before breaking, ensure it's reflected:
            current_selected_content = final_output_state["error_message"] # type: ignore
            titles_for_radio = [] # Clear titles on error
            processed_docs_for_browser = [] # Clear docs on error
        elif not final_output_state.get("retrieved_docs_for_browser"): # If loop finished but no docs processed (e.g. RAG returned empty/bad format)
            current_selected_content = "No documents found or processed."
            titles_for_radio = []
            processed_docs_for_browser = [] # Ensure it's an empty list

        # Final yield outside the loop to ensure UI is updated with the terminal state.
        # This covers cases where the loop finishes without errors, or an error broke the loop.
        yield (titles_for_radio, current_selected_content, _serialize_state_to_str(final_output_state), processed_docs_for_browser)

    except Exception as e:
        print(f"Error during 'Search Docs' graph execution: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        # Update the existing final_output_state if it's a dict, otherwise create a new one.
        if isinstance(final_output_state, dict):
             final_output_state["error_message"] = error_msg # type: ignore
        else: # Should not happen if initial_state is always a GraphState dict
            final_output_state = GraphState(query=user_query, rag_response={}, llm_response={}, error_message=error_msg, fail_name=None, fa_report_name=None, retrieved_docs_for_browser=None, selected_doc_content=None)

        yield ([], error_msg, _serialize_state_to_str(final_output_state), None)

async def run_fa_report_flow(fail_name_input: str, fa_report_name_input: str) -> AsyncGenerator[tuple[str, str], None]:
    """
    Runs the 'Create FA Report' flow (RAG + LLM) and streams outputs for Gradio.
    Yields a tuple: (fa_report_string, serialized_graph_state_string).
    """
    if not fail_name_input or not fail_name_input.strip() or \
       not fa_report_name_input or not fa_report_name_input.strip():
        empty_state_info = GraphState(query=None, rag_response={}, llm_response={}, error_message="Missing inputs", fail_name=fail_name_input, fa_report_name=fa_report_name_input, retrieved_docs_for_browser=None, selected_doc_content=None)
        yield ("Please provide both Failure Name and FA Report Name.", _serialize_state_to_str(empty_state_info))
        return

    initial_state = GraphState(
        query=None, # RAG node will derive query from fail_name
        rag_response={},
        llm_response={},
        error_message=None,
        fail_name=fail_name_input,
        fa_report_name=fa_report_name_input,
        retrieved_docs_for_browser=None,
        selected_doc_content=None
    )

    yield (f"Starting FA Report generation for '{fail_name_input}'...\n---\n", _serialize_state_to_str(initial_state))

    current_report_output = ""
    final_node_output_state: GraphState = initial_state # To hold the state from the last processed node

    try:
        async for event in FA_REPORT_GRAPH.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            node_name = event["name"]
            data = event.get("data", {})

            if isinstance(data.get("input"), dict):
                current_event_input_state_dict = data["input"]
                final_node_output_state = GraphState(
                    query=current_event_input_state_dict.get("query", initial_state.get("query")), # type: ignore
                    rag_response=current_event_input_state_dict.get("rag_response", initial_state.get("rag_response", {})), # type: ignore
                    llm_response=current_event_input_state_dict.get("llm_response", initial_state.get("llm_response", {})), # type: ignore
                    error_message=current_event_input_state_dict.get("error_message", initial_state.get("error_message")), # type: ignore
                    fail_name=current_event_input_state_dict.get("fail_name", initial_state.get("fail_name")), # type: ignore
                    fa_report_name=current_event_input_state_dict.get("fa_report_name", initial_state.get("fa_report_name")), # type: ignore
                    retrieved_docs_for_browser=current_event_input_state_dict.get("retrieved_docs_for_browser", initial_state.get("retrieved_docs_for_browser")), # type: ignore
                    selected_doc_content=current_event_input_state_dict.get("selected_doc_content", initial_state.get("selected_doc_content")) # type: ignore
                )

            if event_type == "on_chain_start":
                if node_name == "retrieve_rag":
                    current_report_output = "Retrieving documents for context...\n"
                elif node_name == "call_llm":
                    current_report_output += "Generating FA Report with LLM...\n"
                yield (current_report_output, _serialize_state_to_str(final_node_output_state))

            elif event_type == "on_chain_end":
                output_state_data = data.get("output")
                if not isinstance(output_state_data, dict): continue

                final_node_output_state = output_state_data.copy() # type: ignore

                error_message = final_node_output_state.get("error_message") # type: ignore
                if error_message:
                    current_report_output += f"Error from {node_name}: {error_message}\n"
                    yield (current_report_output, _serialize_state_to_str(final_node_output_state)) # type: ignore
                    return

                if node_name == "retrieve_rag":
                    current_report_output = "Documents retrieved.\n"

                elif node_name == "call_llm":
                    llm_resp = final_node_output_state.get("llm_response") # type: ignore
                    if llm_resp:
                        try:
                            final_answer = llm_resp.get("choices", [{}])[0].get("message", {}).get("content", "") # type: ignore
                            if final_answer:
                                current_report_output += f"\n--- FA Report for {fa_report_name_input} ---\n{final_answer}\n"
                            else:
                                current_report_output += "LLM generated an empty report or content not found.\n"
                        except (IndexError, AttributeError, TypeError) as e:
                            current_report_output += f"Error parsing LLM response for FA report: {e}. Response: {llm_resp}\n"
                            print(f"Problematic LLM response structure for FA report: {llm_resp}")
                    else:
                        current_report_output += "LLM response missing in the final state.\n"

                yield (current_report_output, _serialize_state_to_str(final_node_output_state)) # type: ignore

        yield (current_report_output, _serialize_state_to_str(final_node_output_state)) # type: ignore

    except Exception as e:
        print(f"Error during 'FA Report' graph execution: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"

        current_state_for_error: GraphState
        if isinstance(final_node_output_state, dict):
            # final_node_output_state["error_message"] = error_msg # This would modify the TypedDict directly, ensure it's allowed or re-construct
            current_state_for_error = final_node_output_state.copy() # type: ignore
            current_state_for_error["error_message"] = error_msg # type: ignore
        else:
            current_state_for_error = GraphState(query=None, rag_response={}, llm_response={}, error_message=error_msg, fail_name=fail_name_input, fa_report_name=fa_report_name_input, retrieved_docs_for_browser=None, selected_doc_content=None)
        yield (error_msg, _serialize_state_to_str(current_state_for_error))

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
    Retrieves documents from the RAG API.
    Uses 'query' if present in state, otherwise constructs one from 'fail_name'.

    Args:
        state: The current graph state.

    Returns:
        The updated graph state with RAG response or an error message.
    """
    current_query = state.get("query")
    fail_name = state.get("fail_name")
    actual_query_for_rag: str | None = None

    if current_query:
        actual_query_for_rag = current_query
        print(f"Node: retrieve_rag (using direct query: {actual_query_for_rag})")
    elif fail_name:
        actual_query_for_rag = f"Detailed information and failure analysis context for: {fail_name}"
        print(f"Node: retrieve_rag (constructing query from fail_name: '{fail_name}' -> '{actual_query_for_rag}')")
        # Update the main query field in the state as well, as it might be used by LLM or for display
        # state = {**state, "query": actual_query_for_rag} # This creates a new dict, be careful with state updates
    else:
        errmsg = "Neither 'query' nor 'fail_name' found in state for RAG retrieval."
        print(f"Node: retrieve_rag ERROR: {errmsg}")
        # Ensure all relevant fields from GraphState are preserved
        return GraphState(
            query=current_query, # or None
            rag_response={},
            llm_response=state.get("llm_response", {}), # Preserve existing
            error_message=errmsg,
            fail_name=fail_name,
            fa_report_name=state.get("fa_report_name"),
            # serialized_graph_state=state.get("serialized_graph_state") # This line should be removed if serialized_graph_state is no longer part of GraphState
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), # Preserve existing
            selected_doc_content=state.get("selected_doc_content") # Preserve existing
        )

    # If fail_name was used to generate query, update the state's query field
    # This is important if the LLM node or UI relies on state['query'] reflecting the RAG query.
    updated_state_query_field = actual_query_for_rag if fail_name and not current_query else current_query

    try:
        rag_api_key = get_env_variable("RAG_API_KEY")
        headers = {**BASE_HEADERS, "Authorization": f"Bearer {rag_api_key}"}
        payload = {**RAG_PAYLOAD_TEMPLATE, "query": actual_query_for_rag} # Use the determined query

        async with httpx.AsyncClient() as client:
            api_response = await _make_api_call(client, f"{RAG_BASE_URL}/query", headers, payload)

        return GraphState(
            query=updated_state_query_field,
            rag_response=api_response,
            llm_response=state.get("llm_response", {}),
            error_message=None,
            fail_name=fail_name,
            fa_report_name=state.get("fa_report_name"),
            # serialized_graph_state=state.get("serialized_graph_state"), # Will be updated by the calling flow
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )
    except ValueError as e: # From get_env_variable
        errmsg = f"RAG API Key error: {e}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=str(e), fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.HTTPStatusError as e_http:
        errmsg = f"RAG API HTTP Status Error: {e_http.response.status_code} - {e_http.response.text}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=f"RAG API Error: {e_http.response.status_code}", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.TimeoutException as e_timeout:
        errmsg = f"RAG API Timeout Error: {e_timeout}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message="RAG API request timed out.", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except Exception as e_generic: # Catch any other unexpected errors
        errmsg = f"An unexpected error occurred in retrieve_rag: {e_generic}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=f"Unexpected RAG error: {type(e_generic).__name__}", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))

async def call_llm(state: GraphState) -> GraphState:
    """
    Calls the LLM API to generate a Failure Analysis report using
    fail_name, fa_report_name, and context from RAG.

    Args:
        state: The current graph state.

    Returns:
        The updated graph state with LLM response or an error message.
    """
    print("Node: call_llm (for FA Report)")

    fail_name = state.get("fail_name")
    fa_report_name = state.get("fa_report_name")
    rag_response = state.get("rag_response")
    # query = state.get("query") # Original query, might be useful for reference or if fail_name is not present

    if not fail_name or not fa_report_name:
        errmsg = "Missing 'fail_name' or 'fa_report_name' in state for LLM report generation."
        print(f"Node: call_llm ERROR: {errmsg}")
        return GraphState(
            query=state.get("query"),
            rag_response=rag_response if rag_response else {},
            llm_response={},
            error_message=errmsg,
            fail_name=fail_name,
            fa_report_name=fa_report_name,
            # serialized_graph_state=state.get("serialized_graph_state") # Removed
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )

    try:
        llm_api_key = get_env_variable("LLM_API_KEY")
        headers = {**BASE_HEADERS, "Authorization": f"Bearer {llm_api_key}"}

        # Prepare context from RAG response
        context_str = "No specific context documents were found or provided."
        if rag_response and isinstance(rag_response.get("documents"), list) and rag_response["documents"]:
            context_items = []
            for doc in rag_response["documents"][:5]: # Use top 5 documents for FA report
                if isinstance(doc, dict) and "text" in doc:
                    context_items.append(doc["text"])
                elif isinstance(doc, str):
                    context_items.append(doc)
            if context_items:
                context_str = "\n\n".join(context_items) # Join with double newlines for better separation

        # Construct the prompt for FA report generation
        prompt_content = (
            f"Please generate a comprehensive Failure Analysis Report.\n\n"
            f"Failure Name/ID: {fail_name}\n"
            f"Report Name/Title: {fa_report_name}\n\n"
            f"Relevant Context from Retrieved Documents:\n--- (BEGIN CONTEXT) ---\n{context_str}\n--- (END CONTEXT) ---\n\n"
            f"Structure the report clearly, including sections like Introduction, Analysis, Findings, Conclusion, and Recommendations if applicable."
        )

        # Using a fresh messages structure, ignoring LLM_PAYLOAD_TEMPLATE's messages for this specific prompt
        messages = [
            {"role": "system", "content": "You are an expert failure analysis report generator."},
            {"role": "user", "content": prompt_content}
        ]

        # Use model from template, but override messages
        payload = {**LLM_PAYLOAD_TEMPLATE, "messages": messages}
        # Remove 'content' from payload if it was part of original template and not messages based
        if "content" in payload and "messages" in LLM_PAYLOAD_TEMPLATE: # if original template had a different structure
            payload.pop("content", None)


        async with httpx.AsyncClient() as client:
            api_response = await _make_api_call(client, f"{LLM_BASE_URL}/chat/completions", headers, payload)

        return GraphState(
            query=state.get("query"), # Preserve original query if any
            rag_response=rag_response if rag_response else {},
            llm_response=api_response,
            error_message=None,
            fail_name=fail_name,
            fa_report_name=fa_report_name,
            # serialized_graph_state=state.get("serialized_graph_state"), # Will be updated by the calling flow
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )
    except ValueError as e: # From get_env_variable
        errmsg = f"LLM API Key error: {e}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=str(e), fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.HTTPStatusError as e_http:
        errmsg = f"LLM API HTTP Status Error: {e_http.response.status_code} - {e_http.response.text}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=f"LLM API Error: {e_http.response.status_code}", fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.TimeoutException as e_timeout:
        errmsg = f"LLM API Timeout Error: {e_timeout}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message="LLM API request timed out.", fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except Exception as e_generic:
        errmsg = f"An unexpected error occurred in call_llm: {e_generic}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=f"Unexpected LLM error: {type(e_generic).__name__}", fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))

def build_search_docs_graph() -> langgraph.graph.CompiledGraph:
    """Builds and compiles a LangGraph for the 'Search Docs' flow (RAG only)."""
    search_graph = StateGraph(GraphState)
    # The function 'retrieve_rag' is added with the name 'retrieve_rag' to this specific graph
    search_graph.add_node("retrieve_rag", retrieve_rag)
    search_graph.set_entry_point("retrieve_rag")
    search_graph.set_finish_point("retrieve_rag")
    compiled_app = search_graph.compile()
    print("Search Docs LangGraph (RAG only) compiled successfully.")
    return compiled_app

SEARCH_DOCS_GRAPH = build_search_docs_graph()

# async def run_graph_for_gradio(user_question: str) -> AsyncGenerator[str, None]:
#     """
#     Runs the LangGraph with the user's question and streams outputs for Gradio.
#     """
#     if not user_question or not user_question.strip():
#         yield "Please enter a question."
#         return
#
#     initial_state = GraphState(
#         query=user_question,
#         rag_response={},
#         llm_response={},
#         error_message=None,
#         fail_name=None,
#         fa_report_name=None,
#         serialized_graph_state=None
#     )
#
#     yield "Starting process...\n---\n"
#     current_output = ""
#
#     try:
#         async for event in FA_REPORT_GRAPH.astream_events(initial_state, version="v1"): # TODO: This needs to be dynamic based on UI mode
#             event_type = event["event"]
#             node_name = event["name"] # Corresponds to node names given in add_node
#             data = event.get("data", {})
#
#             # Useful for debugging event stream:
#             # print(f"Event: {event_type}, Node: {node_name}, Data Keys: {data.keys()}")
#
#             if event_type == "on_chain_start":
#                 if node_name == "retrieve_rag":
#                     current_output = "Retrieving documents...\n"
#                     yield current_output
#                 elif node_name == "call_llm":
#                     current_output += "Calling LLM...\n"
#                     yield current_output
#
#             elif event_type == "on_chain_end":
#                 # 'data' for on_chain_end contains 'output', which is the state *after* the node ran
#                 output_state = data.get("output")
#                 if not isinstance(output_state, dict): # Should be our GraphState dict
#                     # This case should ideally not happen with proper graph setup
#                     print(f"Warning: Output from node {node_name} is not a dict: {output_state}")
#                     continue
#
#                 error_message = output_state.get("error_message")
#                 if error_message:
#                     current_output += f"Error from {node_name}: {error_message}\n"
#                     yield current_output
#                     return # Stop generation on error
#
#                 if node_name == "retrieve_rag":
#                     rag_resp = output_state.get("rag_response")
#                     if rag_resp:
#                         # You could add details from rag_resp if needed
#                         current_output = "Documents retrieved successfully.\n" # Reset for next step
#                         yield current_output
#                     # If RAG failed, error_message above would have caught it.
#
#                 elif node_name == "call_llm":
#                     llm_resp = output_state.get("llm_response")
#                     if llm_resp:
#                         try:
#                             # Adjust this path based on the actual LLM API response structure
#                             # Example: {"choices": [{"message": {"content": "..."}}]}
#                             final_answer = llm_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
#                             if final_answer:
#                                 current_output += f"\nLLM Response:\n{final_answer}\n"
#                             else:
#                                 current_output += "LLM response content not found in the expected structure.\n"
#                             yield current_output
#                         except (IndexError, AttributeError, TypeError) as e:
#                             current_output += f"Error parsing LLM response: {e}. Response: {llm_resp}\n"
#                             yield current_output
#                             print(f"Problematic LLM response structure: {llm_resp}")
#                     # If LLM failed, error_message would have caught it.
#
#         current_output += "\n---\nProcess finished."
#         yield current_output
#
#     except Exception as e:
#         # Catch-all for unexpected errors during graph streaming
#         print(f"Error during graph execution for Gradio: {e}")
#         yield f"\nAn unexpected error occurred: {e}"

# Construct the LangGraph for FA Reports
def build_fa_report_graph() -> langgraph.graph.CompiledGraph:
    """Builds and compiles the LangGraph application for FA Reports."""
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
    print("FA Report LangGraph compiled successfully.")
    return compiled_app

# Create the compiled app instance globally or pass it around
# For simplicity in Gradio, we can make it global or initialize in Gradio setup
# However, calling build_graph() each time Gradio runs a function is also an option,
# though less efficient if graph construction is heavy. Let's make it global for now.

FA_REPORT_GRAPH = build_fa_report_graph()

def on_doc_title_select(selected_title: str, all_retrieved_docs: List[Dict[str, str]] | None) -> str:
    """
    Callback function for when a document title is selected in the Radio list.
    Finds the selected document's content from the stored list of all documents.

    Args:
        selected_title: The title of the document selected via the gr.Radio component.
        all_retrieved_docs: The full list of documents (each a dict with "title" and "content")
                              retrieved from the gr.State component.

    Returns:
        The content of the selected document as a string.
    """
    if not selected_title:
        return "No title selected or title is empty."

    if not all_retrieved_docs:
        # This case might happen if the gr.State is empty or not populated yet
        return "No documents available to select from. Please perform a search first."

    for doc in all_retrieved_docs:
        if doc.get("title") == selected_title:
            return doc.get("content", "Content not found for this title.")

    # Fallback if title not found in the list, though this shouldn't ideally occur
    # if titles for Radio are sourced directly from all_retrieved_docs.
    return f"Error: Content for '{selected_title}' not found in the provided document list."

def launch_gradio_interface():
    """
    Launches the Gradio web interface with tabbed flows and shared state display.
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph RAG/LLM Processor") as demo:
        gr.Markdown("# LangGraph RAG & LLM Processor")

        # Shared component for graph state display - defined once
        # It will be updated by the outputs of the functions called by tabs.
        shared_graph_state_display = gr.Code(
            label="Current LangGraph State Snapshot",
            language="json",
            lines=15,
            interactive=False,
            # value="Graph state will appear here after a run." # Initial value
        )

        with gr.Tabs():
            with gr.TabItem("Search Documents (RAG only)"):
                gr.Markdown("### Document Search & Browser\nSearch for documents using a query. Select a title from the retrieved list to view its content.")
                with gr.Row():
                    search_query_input_db = gr.Textbox(label="Document Search Query", placeholder="Enter query to find relevant documents...", lines=2, scale=3)
                    search_submit_button_db = gr.Button("Search Documents", scale=1)

                with gr.Row():
                    doc_titles_radio_db = gr.Radio(
                        label="Retrieved Document Titles (Top 10)",
                        choices=[], # Initially empty, will be populated by run_search_docs_flow
                        scale=1,
                        elem_id="doc_titles_radio" # Optional: for specific styling/JS
                    )
                    doc_content_display_db = gr.Markdown(
                        label="Selected Document Content",
                        scale=3,
                        value="*Document content will appear here after searching and selecting a title.*" # Initial placeholder
                    )

                # Hidden gr.State component to store the full list of retrieved documents
                # This allows on_doc_title_select to access all document data without another API call.
                all_retrieved_docs_state_db = gr.State([])

                # Event handler for the search button
                search_submit_button_db.click(
                    fn=run_search_docs_flow,
                    inputs=[search_query_input_db],
                    outputs=[
                        doc_titles_radio_db,          # Output 1: List of titles for Radio
                        doc_content_display_db,       # Output 2: Content of the first document
                        shared_graph_state_display,   # Output 3: Serialized graph state (already defined)
                        all_retrieved_docs_state_db   # Output 4: Full list of docs for gr.State
                    ]
                )

                # Event handler for when a title is selected in the Radio component
                doc_titles_radio_db.select(
                    fn=on_doc_title_select,
                    inputs=[doc_titles_radio_db, all_retrieved_docs_state_db], # Current radio value, all docs from state
                    outputs=[doc_content_display_db]                          # Update the Markdown content display
                )

            with gr.TabItem("Create FA Report (RAG + LLM)"):
                with gr.Row():
                    fa_fail_name_input = gr.Textbox(label="Failure Name / ID", placeholder="e.g., Component_X_Failure_001")
                    fa_report_name_input = gr.Textbox(label="FA Report Name / Title", placeholder="e.g., FA Report for Component_X_Failure_001")
                fa_submit_button = gr.Button("Generate FA Report")
                fa_report_output = gr.Markdown(label="Generated Failure Analysis Report") # Markdown for report

                fa_submit_button.click(
                    fn=run_fa_report_flow,
                    inputs=[fa_fail_name_input, fa_report_name_input],
                    outputs=[fa_report_output, shared_graph_state_display] # Tuple yielded by fn maps here
                )

        gr.Markdown(
            "**Note**: API keys for RAG and LLM services must be set as "
            "environment variables (`RAG_API_KEY`, `LLM_API_KEY`). "
            "The full state displayed is a snapshot and might be truncated for readability."
        )

    print("Launching Gradio interface with Tabs...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

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
