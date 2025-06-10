import asyncio
import os
from typing import TypedDict, List, Dict, Any, AsyncGenerator

import json
import httpx
import gradio as gr
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langgraph.graph import StateGraph

LLM_BASE_URL = "http://ho1lee.aabb.cc"
RAG_BASE_URL = "http://ho2lee.aabb.cc"
LLM_PAYLOAD_TEMPLATE = {"model": "mistral-7b", "messages": [{"role": "user", "content": ""}]}
RAG_PAYLOAD_TEMPLATE = {"query": "", "top_k": 5}
BASE_HEADERS = {"Content-Type": "application/json"}

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        failure_summary: Text content from the global Failure Summary textbox.
        query: The user's input query, potentially derived from failure_summary or specific inputs.
        rag_response: The response from the RAG API.
        llm_response: The response from the LLM API.
        error_message: Optional error message if a step fails.
        fail_name: Specific name/ID for FA report.
        fa_report_name: Title for FA report.
        retrieved_docs_for_browser: List of documents with title and content for UI browser.
        selected_doc_content: Content of the currently selected document in UI browser.
    """
    failure_summary: str | None
    query: str | None
    rag_response: Dict[str, Any]
    llm_response: Dict[str, Any]
    error_message: str | None
    fail_name: str | None
    fa_report_name: str | None
    retrieved_docs_for_browser: List[Dict[str, str]] | None
    selected_doc_content: str | None

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not set. Please ensure it is available.")
    return value

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    reraise=True
)
async def _make_api_call(client: httpx.AsyncClient, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    response = await client.post(url, json=payload, headers=headers, timeout=10.0)
    response.raise_for_status()
    return response.json()

def _serialize_state_to_str(state: GraphState) -> str:
    display_state = {k: v for k, v in state.items() if k not in ["rag_response", "llm_response", "serialized_graph_state"]}
    if state.get("rag_response") and isinstance(state.get("rag_response"), dict):
        display_state["rag_summary"] = {
            "document_count": len(state["rag_response"].get("documents", [])),
            "keys": list(state["rag_response"].keys())
        }
    if state.get("llm_response") and isinstance(state.get("llm_response"), dict):
        display_state["llm_summary"] = {
            "keys": list(state["llm_response"].keys()),
            "choice_1_content_preview": state["llm_response"].get("choices", [{}])[0].get("message",{}).get("content","")[:100]+"..."
        }
    return json.dumps(display_state, indent=2, default=str)

async def run_search_docs_flow(failure_summary_from_global_input: str) -> AsyncGenerator[tuple[Any, str, str, List[Dict[str, str]] | None], None]:
    initial_state = GraphState(
        failure_summary=failure_summary_from_global_input,
        query=failure_summary_from_global_input,
        rag_response={}, llm_response={}, error_message=None,
        fail_name=None, fa_report_name=None,
        retrieved_docs_for_browser=None, selected_doc_content=None
    )
    if not failure_summary_from_global_input or not failure_summary_from_global_input.strip():
        no_query_msg = "Please enter a Failure Summary to search for documents."
        error_state = GraphState(
            failure_summary=failure_summary_from_global_input, query=None,
            rag_response={}, llm_response={}, error_message=no_query_msg,
            fail_name=None, fa_report_name=None,
            retrieved_docs_for_browser=None, selected_doc_content=None
        )
        yield (gr.update(choices=[], value=None), no_query_msg, _serialize_state_to_str(error_state), [])
        return
    yield (gr.update(choices=[], value=None), "Searching...", _serialize_state_to_str(initial_state), None)
    processed_docs_list: List[Dict[str, str]] = []
    titles_list: List[str] = []
    selected_content: str = "Processing..."
    current_graph_state: GraphState = initial_state
    data_processed_for_node = False
    try:
        async for event in SEARCH_DOCS_GRAPH.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            node_name = event["name"]
            data = event.get("data", {})
            if event_type == "on_chain_start":
                data_processed_for_node = False
                if node_name == "retrieve_rag":
                    selected_content = "Retrieving documents..."
                    yield (gr.update(choices=[], value=None), selected_content, _serialize_state_to_str(current_graph_state), processed_docs_list)
            elif event_type == "on_chain_stream":
                chunk = data.get("chunk")
                if isinstance(chunk, dict) and node_name == "retrieve_rag":
                    current_graph_state = chunk.copy()
                    data_processed_for_node = True
                    error_msg_from_state = current_graph_state.get("error_message")
                    if error_msg_from_state:
                        selected_content = f"Error: {error_msg_from_state}"
                        titles_list = []
                        processed_docs_list = []
                        yield (gr.update(choices=[], value=None), selected_content, _serialize_state_to_str(current_graph_state), processed_docs_list)
                        return
                    raw_docs = current_graph_state.get("rag_response", {}).get("documents", [])
                    processed_docs_list = []
                    for i, doc_data in enumerate(raw_docs[:10]):
                        if not isinstance(doc_data, dict): continue
                        _content_val = doc_data.get("content", doc_data.get("text", ""))
                        title = doc_data.get("title", doc_data.get("name", _content_val[:50] + "..." if _content_val else f"Document {i+1}"))
                        content = _content_val if _content_val else "Content not available."
                        processed_docs_list.append({"title": title, "content": content})
                    current_graph_state["retrieved_docs_for_browser"] = processed_docs_list
                    if processed_docs_list:
                        titles_list = [doc["title"] for doc in processed_docs_list]
                        selected_content = processed_docs_list[0]["content"]
                        current_graph_state["selected_doc_content"] = selected_content
                    else:
                        titles_list = []
                        selected_content = "No documents found."
                        current_graph_state["selected_doc_content"] = selected_content
                    yield (gr.update(choices=titles_list, value=titles_list[0] if titles_list else None), selected_content, _serialize_state_to_str(current_graph_state), processed_docs_list)
            elif event_type == "on_chain_end":
                if node_name == "retrieve_rag":
                    if not data_processed_for_node and isinstance(data.get("output"), dict) :
                        print(f"Note: on_chain_end for {node_name} reached, data_processed_for_node is False.")
                    yield (gr.update(choices=titles_list, value=titles_list[0] if titles_list else None), selected_content, _serialize_state_to_str(current_graph_state), processed_docs_list)
        yield (gr.update(choices=titles_list, value=titles_list[0] if titles_list else None), selected_content, _serialize_state_to_str(current_graph_state), processed_docs_list)
    except Exception as e:
        print(f"Error in run_search_docs_flow: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        if not isinstance(current_graph_state, dict): current_graph_state = initial_state.copy()
        current_graph_state["error_message"] = error_msg
        current_graph_state["failure_summary"] = failure_summary_from_global_input
        yield (gr.update(choices=[], value=None), error_msg, _serialize_state_to_str(current_graph_state), None)

async def run_fa_report_flow(fail_name_input: str, fa_report_name_input: str, global_failure_summary: str | None) -> AsyncGenerator[tuple[str, str], None]:
    initial_state = GraphState(
        failure_summary=global_failure_summary,
        query=None, rag_response={}, llm_response={}, error_message=None,
        fail_name=fail_name_input, fa_report_name=fa_report_name_input,
        retrieved_docs_for_browser=None, selected_doc_content=None
    )
    if not fail_name_input or not fail_name_input.strip() or not fa_report_name_input or not fa_report_name_input.strip():
        initial_state["error_message"] = "Missing inputs for FA Report Name or Failure Name/ID."
        yield ("Please provide both Failure Name/ID and FA Report Name.", _serialize_state_to_str(initial_state))
        return
    current_report_output = f"Starting FA Report generation for '{fail_name_input}'...\n---\n"
    yield (current_report_output, _serialize_state_to_str(initial_state))
    current_graph_state: GraphState = initial_state
    data_processed_for_node = False
    rag_completed_successfully = False
    try:
        async for event in FA_REPORT_GRAPH.astream_events(initial_state, version="v1"):
            event_type = event["event"]
            node_name = event["name"]
            data = event.get("data", {})
            if isinstance(data.get("input"), dict):
                current_event_input_state_dict = data["input"]
                current_graph_state = GraphState(
                    failure_summary=current_event_input_state_dict.get("failure_summary", initial_state.get("failure_summary")),
                    query=current_event_input_state_dict.get("query", initial_state.get("query")),
                    rag_response=current_event_input_state_dict.get("rag_response", initial_state.get("rag_response", {})),
                    llm_response=current_event_input_state_dict.get("llm_response", initial_state.get("llm_response", {})),
                    error_message=current_event_input_state_dict.get("error_message", initial_state.get("error_message")),
                    fail_name=current_event_input_state_dict.get("fail_name", initial_state.get("fail_name")),
                    fa_report_name=current_event_input_state_dict.get("fa_report_name", initial_state.get("fa_report_name")),
                    retrieved_docs_for_browser=current_event_input_state_dict.get("retrieved_docs_for_browser", initial_state.get("retrieved_docs_for_browser")),
                    selected_doc_content=current_event_input_state_dict.get("selected_doc_content", initial_state.get("selected_doc_content"))
                )
            if event_type == "on_chain_start":
                data_processed_for_node = False
                if node_name == "retrieve_rag":
                    current_report_output = "Retrieving documents for context...\n"
                elif node_name == "call_llm":
                    current_report_output = ("Documents retrieved successfully.\n" if rag_completed_successfully else "") + "Generating FA Report with LLM...\n"
                yield (current_report_output, _serialize_state_to_str(current_graph_state))
            elif event_type == "on_chain_stream":
                chunk = data.get("chunk")
                if isinstance(chunk, dict):
                    current_graph_state = chunk.copy() # type: ignore
                    data_processed_for_node = True
                    error_msg_from_state = current_graph_state.get("error_message")
                    if error_msg_from_state:
                        current_report_output += f"Error from {node_name}: {error_msg_from_state}\n"
                        yield (current_report_output, _serialize_state_to_str(current_graph_state))
                        return
                    if node_name == "retrieve_rag":
                        rag_completed_successfully = True
                        yield (current_report_output, _serialize_state_to_str(current_graph_state))
                    elif node_name == "call_llm":
                        llm_resp = current_graph_state.get("llm_response")
                        temp_report_addition = ""
                        if llm_resp:
                            try:
                                final_answer = llm_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                                if final_answer: temp_report_addition = f"\n--- FA Report for {fa_report_name_input} ---\n{final_answer}\n"
                                else: temp_report_addition = "LLM generated an empty report or content not found.\n"
                            except (IndexError, AttributeError, TypeError) as e:
                                temp_report_addition = f"Error parsing LLM response: {e}. Response: {llm_resp}\n"
                        else: temp_report_addition = "LLM response missing in the final state.\n"
                        base_output = "Documents retrieved successfully.\n" if rag_completed_successfully else ""
                        base_output += "Generating FA Report with LLM...\n"
                        current_report_output = base_output + temp_report_addition
                        yield (current_report_output, _serialize_state_to_str(current_graph_state))
            elif event_type == "on_chain_end":
                if not data_processed_for_node and isinstance(data.get("output"), dict):
                    current_graph_state = data["output"].copy() # type: ignore
                    print(f"Warning: Data for {node_name} processed from on_chain_end for FA report.")
                if node_name == "call_llm":
                    yield (current_report_output, _serialize_state_to_str(current_graph_state))
                elif node_name == "retrieve_rag" and data_processed_for_node:
                     current_report_output = "Documents retrieved successfully.\n"
                     if "Generating FA Report with LLM" in current_report_output :
                         current_report_output += "Generating FA Report with LLM...\n"
                     yield (current_report_output, _serialize_state_to_str(current_graph_state))
        yield (current_report_output, _serialize_state_to_str(current_graph_state))
    except Exception as e:
        print(f"Error in run_fa_report_flow: {e}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        if not isinstance(current_graph_state, dict): current_graph_state = initial_state.copy() # type: ignore
        current_graph_state["error_message"] = error_msg
        current_graph_state["failure_summary"] = initial_state.get("failure_summary")
        yield (error_msg, _serialize_state_to_str(current_graph_state))

async def retrieve_rag(state: GraphState) -> GraphState:
    current_query = state.get("query")
    fail_name = state.get("fail_name")
    failure_summary_from_state = state.get("failure_summary")
    actual_query_for_rag: str | None = None
    query_was_generated_for_fa = False
    if fail_name:
        query_was_generated_for_fa = True
        base_query_part = f"Gather detailed information for failure ID/name: '{fail_name}'."
        summary_part = f"Additional context/summary: '{failure_summary_from_state if failure_summary_from_state else 'Not provided.'}'"
        instruction_part = "Focus on technical specifications, previous similar incidents, and troubleshooting guides."
        actual_query_for_rag = f"{base_query_part} {summary_part} {instruction_part}"
        print(f"Node: retrieve_rag (FA Flow Query): {actual_query_for_rag}")
    elif current_query:
        actual_query_for_rag = current_query
        print(f"Node: retrieve_rag (Search Docs Flow Query): {actual_query_for_rag}")
    else:
        errmsg = "For RAG: provide 'fail_name' (for FA flow) or 'query' (for Search flow, from global summary)."
        print(f"Node: retrieve_rag ERROR: {errmsg}")
        return GraphState(
            failure_summary=failure_summary_from_state, query=current_query,
            rag_response={}, llm_response=state.get("llm_response", {}), error_message=errmsg,
            fail_name=fail_name, fa_report_name=state.get("fa_report_name"),
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )
    output_state_query_field: str | None
    if query_was_generated_for_fa: output_state_query_field = actual_query_for_rag
    else: output_state_query_field = current_query
    try:
        rag_api_key = get_env_variable("RAG_API_KEY")
        headers = {**BASE_HEADERS, "Authorization": f"Bearer {rag_api_key}"}
        payload = {**RAG_PAYLOAD_TEMPLATE, "query": actual_query_for_rag}
        async with httpx.AsyncClient() as client:
            api_response = await _make_api_call(client, f"{RAG_BASE_URL}/query", headers, payload)
        return GraphState(
            failure_summary=failure_summary_from_state, query=output_state_query_field,
            rag_response=api_response, llm_response=state.get("llm_response", {}), error_message=None,
            fail_name=fail_name, fa_report_name=state.get("fa_report_name"),
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )
    except ValueError as e:
        errmsg = f"RAG API Key error: {e}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=output_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=str(e), fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.HTTPStatusError as e_http:
        errmsg = f"RAG API HTTP Status Error: {e_http.response.status_code} - {e_http.response.text}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=output_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=f"RAG API Error: {e_http.response.status_code}", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.TimeoutException as e_timeout:
        errmsg = f"RAG API Timeout Error: {e_timeout}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=output_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message="RAG API request timed out.", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except Exception as e_generic:
        errmsg = f"An unexpected error occurred in retrieve_rag: {e_generic}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=output_state_query_field, rag_response={}, llm_response=state.get("llm_response", {}), error_message=f"Unexpected RAG error: {type(e_generic).__name__}", fail_name=fail_name, fa_report_name=state.get("fa_report_name"), retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))

async def call_llm(state: GraphState) -> GraphState:
    print("Node: call_llm (for FA Report)")
    fail_name = state.get("fail_name")
    fa_report_name = state.get("fa_report_name")
    rag_response = state.get("rag_response")
    failure_summary_from_state = state.get("failure_summary")
    if not fail_name or not fa_report_name:
        errmsg = "Missing 'fail_name' or 'fa_report_name' in state for LLM report generation."
        print(f"Node: call_llm ERROR: {errmsg}")
        return GraphState(
            failure_summary=failure_summary_from_state, query=state.get("query"),
            rag_response=rag_response if rag_response else {}, llm_response={}, error_message=errmsg,
            fail_name=fail_name, fa_report_name=fa_report_name,
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )
    try:
        llm_api_key = get_env_variable("LLM_API_KEY")
        headers = {**BASE_HEADERS, "Authorization": f"Bearer {llm_api_key}"}
        context_str = "No specific context documents were found or provided."
        if rag_response and isinstance(rag_response.get("documents"), list) and rag_response["documents"]:
            context_items = []
            for doc in rag_response["documents"][:5]:
                if isinstance(doc, dict) and "text" in doc: context_items.append(doc["text"])
                elif isinstance(doc, str): context_items.append(doc)
            if context_items: context_str = "\n\n".join(context_items)
        prompt_content = (
            f"Please generate a comprehensive Failure Analysis Report based on all the following information.\n\n"
            f"Overall Failure Summary: {failure_summary_from_state if failure_summary_from_state else 'Not provided.'}\n"
            f"Specific Failure Name/ID: {fail_name}\n"
            f"Intended Report Name/Title: {fa_report_name}\n\n"
            f"Relevant Context from Retrieved Documents:\n--- (BEGIN CONTEXT) ---\n{context_str}\n--- (END CONTEXT) ---\n\n"
            f"Please structure the report clearly and professionally, including sections such as Introduction, Detailed Analysis of Failure, Key Findings, Conclusion, and Recommendations for corrective actions, if applicable."
        )
        messages = [
            {"role": "system", "content": "You are an expert failure analysis report generator."},
            {"role": "user", "content": prompt_content}
        ]
        payload = {**LLM_PAYLOAD_TEMPLATE, "messages": messages}
        if "content" in payload and "messages" in LLM_PAYLOAD_TEMPLATE: payload.pop("content", None)
        async with httpx.AsyncClient() as client:
            api_response = await _make_api_call(client, f"{LLM_BASE_URL}/chat/completions", headers, payload)
        return GraphState(
            failure_summary=failure_summary_from_state, query=state.get("query"),
            rag_response=rag_response if rag_response else {}, llm_response=api_response, error_message=None,
            fail_name=fail_name, fa_report_name=fa_report_name,
            retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"),
            selected_doc_content=state.get("selected_doc_content")
        )
    except ValueError as e:
        errmsg = f"LLM API Key error: {e}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=str(e), fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.HTTPStatusError as e_http:
        errmsg = f"LLM API HTTP Status Error: {e_http.response.status_code} - {e_http.response.text}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=f"LLM API Error: {e_http.response.status_code}", fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except httpx.TimeoutException as e_timeout:
        errmsg = f"LLM API Timeout Error: {e_timeout}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message="LLM API request timed out.", fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))
    except Exception as e_generic:
        errmsg = f"An unexpected error occurred in call_llm: {e_generic}"
        print(errmsg)
        return GraphState(failure_summary=failure_summary_from_state, query=state.get("query"), rag_response=rag_response if rag_response else {}, llm_response={}, error_message=f"Unexpected LLM error: {type(e_generic).__name__}", fail_name=fail_name, fa_report_name=fa_report_name, retrieved_docs_for_browser=state.get("retrieved_docs_for_browser"), selected_doc_content=state.get("selected_doc_content"))

def build_search_docs_graph() -> langgraph.graph.CompiledGraph:
    search_graph = StateGraph(GraphState)
    search_graph.add_node("retrieve_rag", retrieve_rag)
    search_graph.set_entry_point("retrieve_rag")
    search_graph.set_finish_point("retrieve_rag")
    compiled_app = search_graph.compile()
    print("Search Docs LangGraph (RAG only) compiled successfully.")
    return compiled_app
SEARCH_DOCS_GRAPH = build_search_docs_graph()

def build_fa_report_graph() -> langgraph.graph.CompiledGraph:
    graph = StateGraph(GraphState)
    graph.add_node("retrieve_rag", retrieve_rag)
    graph.add_node("call_llm", call_llm)
    graph.set_entry_point("retrieve_rag")
    graph.add_edge("retrieve_rag", "call_llm")
    graph.set_finish_point("call_llm")
    compiled_app = graph.compile()
    print("FA Report LangGraph compiled successfully.")
    return compiled_app
FA_REPORT_GRAPH = build_fa_report_graph()

def on_doc_title_select(selected_title: str, all_retrieved_docs: List[Dict[str, str]] | None) -> str:
    if not selected_title: return "No title selected or title is empty."
    if not all_retrieved_docs: return "No documents available to select from. Please perform a search first."
    for doc in all_retrieved_docs:
        if doc.get("title") == selected_title:
            return doc.get("content", "Content not found for this title.")
    return f"Error: Content for '{selected_title}' not found in the provided document list."

def launch_gradio_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph RAG/LLM Processor") as demo:
        gr.Markdown("# LangGraph RAG & LLM Processor")
        global_failure_summary_input = gr.Textbox(
            label="Global Failure Summary (Context for all operations)",
            placeholder="Enter a general summary of the failure or context here. This will be used if specific queries are not provided in tabs.",
            lines=3,
        )
        shared_graph_state_display = gr.Code(
            label="Current LangGraph State Snapshot", language="json", lines=15, interactive=False,
        )
        with gr.Tabs():
            with gr.TabItem("Search Documents (RAG only)"):
                gr.Markdown("### Document Search & Browser\nSearch for documents using the Global Failure Summary. Select a title from the retrieved list to view its content.")
                search_submit_button_db = gr.Button("Search Documents (using Global Summary)")
                with gr.Row():
                    doc_titles_radio_db = gr.Radio(label="Retrieved Document Titles (Top 10)", choices=[], scale=1, elem_id="doc_titles_radio")
                    doc_content_display_db = gr.Markdown(label="Selected Document Content", value="*Document content will appear here...*", scale=3)
                all_retrieved_docs_state_db = gr.State([])
                search_submit_button_db.click(
                    fn=run_search_docs_flow, inputs=[global_failure_summary_input],
                    outputs=[doc_titles_radio_db, doc_content_display_db, shared_graph_state_display, all_retrieved_docs_state_db]
                )
                doc_titles_radio_db.select(
                    fn=on_doc_title_select, inputs=[doc_titles_radio_db, all_retrieved_docs_state_db], outputs=[doc_content_display_db]
                )
            with gr.TabItem("Create FA Report (RAG + LLM)"):
                with gr.Row():
                    fa_fail_name_input = gr.Textbox(label="Failure Name / ID", placeholder="e.g., Component_X_Failure_001")
                    fa_report_name_input = gr.Textbox(label="FA Report Name / Title", placeholder="e.g., FA Report for Component_X_Failure_001")
                fa_submit_button = gr.Button("Generate FA Report")
                fa_report_output = gr.Markdown(label="Generated Failure Analysis Report")
                fa_submit_button.click(
                    fn=run_fa_report_flow, inputs=[fa_fail_name_input, fa_report_name_input, global_failure_summary_input],
                    outputs=[fa_report_output, shared_graph_state_display]
                )
        gr.Markdown(
            "**Note**: API keys for RAG and LLM services must be set as environment variables (`RAG_API_KEY`, `LLM_API_KEY`). "
            "The full state displayed is a snapshot and might be truncated for readability."
        )
    print("Launching Gradio interface with Tabs...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    try:
        print("Checking for necessary API keys...")
        get_env_variable("LLM_API_KEY")
        get_env_variable("RAG_API_KEY")
        print("API keys found. Proceeding to launch Gradio interface.")
        launch_gradio_interface()
    except ValueError as e:
        print(f"Startup Error: {e}")
        print("Please set the required environment variables before running the application.")
    except Exception as e:
        print(f"An unexpected error occurred during startup: {e}")

```
