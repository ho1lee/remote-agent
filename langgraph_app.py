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

        # Field for displaying the full state in the UI
        serialized_graph_state: str | None
    """
    query: str | None # Query can be None if fail_name is used
    rag_response: Dict[str, Any]
    llm_response: Dict[str, Any]
    error_message: str | None

    # Fields for "Create FA Report" flow
    fail_name: str | None
    fa_report_name: str | None

    # Field for displaying the full state in the UI
    # serialized_graph_state: str | None # Removed as per new design

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

async def run_search_docs_flow(user_query: str) -> AsyncGenerator[tuple[str, str], None]:
    """
    Runs the 'Search Docs' flow (RAG only) and streams outputs for Gradio.
    Yields a tuple: (documents_display_string, serialized_graph_state_string).
    """
    if not user_query or not user_query.strip():
        # Create a valid GraphState without serialized_graph_state for serialization
        empty_state = GraphState(query=None, rag_response={}, llm_response={}, error_message="No query", fail_name=None, fa_report_name=None)
        yield ("Please enter a query to search.", _serialize_state_to_str(empty_state))
        return

    initial_state = GraphState(
        query=user_query,
        rag_response={},
        llm_response={}, # Not used in this flow, but part of state
        error_message=None,
        fail_name=None,
        fa_report_name=None
        # serialized_graph_state=None # Removed
    )

    yield ("Starting document search...\n---\n", _serialize_state_to_str(initial_state))

    current_docs_output = ""
    final_state_for_display = initial_state

    try:
        async for event in SEARCH_DOCS_GRAPH.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            node_name = event["name"]
            data = event.get("data", {})

            if event_type == "on_chain_start":
                if node_name == "retrieve_rag":
                    current_docs_output = "Retrieving documents...\n"
                    yield (current_docs_output, _serialize_state_to_str(initial_state)) # Show initial state during run

            elif event_type == "on_chain_end":
                output_state = data.get("output")
                if not isinstance(output_state, dict): continue

                final_node_output_state = output_state.copy() # Keep a reference to the final state # type: ignore

                error_message = output_state.get("error_message") # type: ignore
                if error_message:
                    current_docs_output += f"Error: {error_message}\n"
                    yield (current_docs_output, _serialize_state_to_str(final_node_output_state)) # type: ignore
                    return

                if node_name == "retrieve_rag":
                    rag_resp = output_state.get("rag_response") # type: ignore
                    if rag_resp and isinstance(rag_resp.get("documents"), list): # type: ignore
                        docs = rag_resp["documents"] # type: ignore
                        if docs:
                            current_docs_output = f"Retrieved {len(docs)} documents:\n---\n"
                            for i, doc_item in enumerate(docs):
                                # Assuming doc_item can be a dict with 'text' or just a string
                                doc_text = doc_item.get("text") if isinstance(doc_item, dict) else str(doc_item)
                                current_docs_output += f"Document {i+1}:\n{doc_text}\n---\n"
                        else:
                            current_docs_output = "No documents found for the query.\n"
                    else:
                        current_docs_output = "RAG response format unexpected or empty.\n"

                    yield (current_docs_output, _serialize_state_to_str(final_node_output_state)) # type: ignore

        # Final yield after loop, just in case (though on_chain_end should be the last)
        # Ensure final state is displayed
        yield (current_docs_output, _serialize_state_to_str(final_node_output_state)) # type: ignore

    except Exception as e:
        print(f"Error during 'Search Docs' graph execution: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        # Ensure final_node_output_state has the error too if possible
        if isinstance(final_node_output_state, dict):
            final_node_output_state["error_message"] = error_msg
        # Create a minimal state if it's not a dict for some reason at this point
        current_state_for_error = final_node_output_state if isinstance(final_node_output_state, dict) else GraphState(query=user_query, rag_response={}, llm_response={}, error_message=error_msg, fail_name=None, fa_report_name=None)
        yield (error_msg, _serialize_state_to_str(current_state_for_error)) # type: ignore

async def run_fa_report_flow(fail_name_input: str, fa_report_name_input: str) -> AsyncGenerator[tuple[str, str], None]:
    """
    Runs the 'Create FA Report' flow (RAG + LLM) and streams outputs for Gradio.
    Yields a tuple: (fa_report_string, serialized_graph_state_string).
    """
    if not fail_name_input or not fail_name_input.strip() or \
       not fa_report_name_input or not fa_report_name_input.strip():
        empty_state_info = GraphState(query=None, rag_response={}, llm_response={}, error_message="Missing inputs", fail_name=fail_name_input, fa_report_name=fa_report_name_input)
        yield ("Please provide both Failure Name and FA Report Name.", _serialize_state_to_str(empty_state_info))
        return

    initial_state = GraphState(
        query=None, # RAG node will derive query from fail_name
        rag_response={},
        llm_response={},
        error_message=None,
        fail_name=fail_name_input,
        fa_report_name=fa_report_name_input
        # serialized_graph_state=None # Removed
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
                    query=current_event_input_state_dict.get("query", initial_state.get("query")),
                    rag_response=current_event_input_state_dict.get("rag_response", initial_state.get("rag_response", {})),
                    llm_response=current_event_input_state_dict.get("llm_response", initial_state.get("llm_response", {})),
                    error_message=current_event_input_state_dict.get("error_message", initial_state.get("error_message")),
                    fail_name=current_event_input_state_dict.get("fail_name", initial_state.get("fail_name")),
                    fa_report_name=current_event_input_state_dict.get("fa_report_name", initial_state.get("fa_report_name"))
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
            current_state_for_error = GraphState(query=None, rag_response={}, llm_response={}, error_message=error_msg, fail_name=fail_name_input, fa_report_name=fa_report_name_input)
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
            serialized_graph_state=state.get("serialized_graph_state")
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
            serialized_graph_state=state.get("serialized_graph_state") # Will be updated by the calling flow
        )
    except ValueError as e: # From get_env_variable
        errmsg = f"RAG API Key error: {e}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=str(e), fail_name=fail_name, fa_report_name=state.get("fa_report_name"), serialized_graph_state=state.get("serialized_graph_state"))
    except httpx.HTTPStatusError as e_http:
        errmsg = f"RAG API HTTP Status Error: {e_http.response.status_code} - {e_http.response.text}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=f"RAG API Error: {e_http.response.status_code}", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), serialized_graph_state=state.get("serialized_graph_state"))
    except httpx.TimeoutException as e_timeout:
        errmsg = f"RAG API Timeout Error: {e_timeout}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message="RAG API request timed out.", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), serialized_graph_state=state.get("serialized_graph_state"))
    except Exception as e_generic: # Catch any other unexpected errors
        errmsg = f"An unexpected error occurred in retrieve_rag: {e_generic}"
        print(errmsg)
        return GraphState(query=updated_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=f"Unexpected RAG error: {type(e_generic).__name__}", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), serialized_graph_state=state.get("serialized_graph_state"))

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
            serialized_graph_state=state.get("serialized_graph_state")
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
            serialized_graph_state=state.get("serialized_graph_state") # Will be updated by the calling flow
        )
    except ValueError as e: # From get_env_variable
        errmsg = f"LLM API Key error: {e}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=str(e), fail_name=fail_name, fa_report_name=fa_report_name, serialized_graph_state=state.get("serialized_graph_state"))
    except httpx.HTTPStatusError as e_http:
        errmsg = f"LLM API HTTP Status Error: {e_http.response.status_code} - {e_http.response.text}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=f"LLM API Error: {e_http.response.status_code}", fail_name=fail_name, fa_report_name=fa_report_name, serialized_graph_state=state.get("serialized_graph_state"))
    except httpx.TimeoutException as e_timeout:
        errmsg = f"LLM API Timeout Error: {e_timeout}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message="LLM API request timed out.", fail_name=fail_name, fa_report_name=fa_report_name, serialized_graph_state=state.get("serialized_graph_state"))
    except Exception as e_generic:
        errmsg = f"An unexpected error occurred in call_llm: {e_generic}"
        print(errmsg)
        return GraphState(query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=f"Unexpected LLM error: {type(e_generic).__name__}", fail_name=fail_name, fa_report_name=fa_report_name, serialized_graph_state=state.get("serialized_graph_state"))

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
                with gr.Row():
                    search_query_input = gr.Textbox(label="Document Search Query", placeholder="Enter query...", lines=3, scale=3)
                    search_submit_button = gr.Button("Search", scale=1)
                search_docs_output = gr.Markdown(label="Retrieved Documents") # Using Markdown for better formatting potential

                search_submit_button.click(
                    fn=run_search_docs_flow,
                    inputs=[search_query_input],
                    outputs=[search_docs_output, shared_graph_state_display] # Tuple yielded by fn maps here
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
