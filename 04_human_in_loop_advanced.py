from typing import Annotated, Dict, Any
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

    
def get_input(interrupt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get user input from the console based on the interrupt data"""

    print("🤖 HUMAN INPUT REQUIRED")
    print("="*60)
    print(f"Question: {interrupt_data.get('query', 'Please review the following information:')}")
    
    if 'information' in interrupt_data:
        print(f"\n{interrupt_data['information']}")
    
    print("\nIs this information correct? (y/n): ", end="")
    correct = input().strip().lower()
    
    if correct.startswith('y'):
        response = {"correct": "yes"}
        print("Information confirmed as correct!")
    else:
        print("\nPlease provide the correct information:")
        correction = input("Enter the correct information: ").strip()
        response = {"correct": "no", "correction": correction}
        print("Correction received!")    
    print("="*60)

    return response


# State for data in the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    human_feedback: str

@tool
def human_assistance(
    information_to_verify: str, tool_call_id: Annotated[str, InjectedToolCallId]
):
    """Request human verification of specific facts or information found from searches.
    
    Use this tool to verify factual information with a human user. Pass the actual 
    facts/data you found as the information_to_verify parameter, NOT a request for the human to find information.
    
    Args:
        information_to_verify: The specific facts, dates, or information you want 
                              the human to verify (e.g., 'iPhone was released on June 29, 2007')
    """
    interrupt_data = {
        "query": "Please review and verify this information:",
        "information": information_to_verify
    }
    
    # This will pause execution and wait for human input
    human_response = interrupt(interrupt_data)
    
    # Process the human response (after getting human input)
    if human_response.get("correct", "").lower().startswith("y"):
        response = "Information verified as correct by user"
        feedback = "User confirmed information is accurate"
    else:
        corrected_info = human_response.get("correction", "No correction provided")
        response = f"CORRECTION: The user corrected the information. The original information '{information_to_verify}' was incorrect. The correct information is: {corrected_info}"
        feedback = f"User corrected the information: {corrected_info}"

    # Update state with verified information
    state_update = {
        "human_feedback": feedback,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)]
    }
    return Command(update=state_update)

def build_graph():
    """Create the interactive graph with human-in-the-loop capabilities"""
    
    # Initialize LLM
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    # Create tools
    search_tool = TavilySearch(max_results=2)
    tools = [search_tool, human_assistance]
    llm_with_tools=llm.bind_tools(tools)

    # graph node
    def chatbot(state: State):
        # Add system message with instructions
        system_message = {
            "role": "system", 
            "content": """You are a helpful AI assistant that searches for information and verifies it with humans.

IMPORTANT INSTRUCTIONS:
1. When asked to search for information, use the search tool to find facts
2. After finding specific information, use the human_assistance tool to verify it with the user
3. Pass the actual facts you found to human_assistance (e.g., "Python was released in February 1991"), NOT a request for the user to find information
4. CRITICAL: After getting human feedback, READ THE TOOL RESPONSE CAREFULLY:
   - If it says "Information verified as correct", use your original facts in the final answer
   - If it starts with "CORRECTION:", the human provided corrections - USE THE CORRECTED INFORMATION in your final answer, NOT your original findings
"""
}
        
        # Prepare messages with system instruction
        messages = [system_message] + state["messages"]
        message = llm_with_tools.invoke(messages)
        # make sure to only call one tool at a time
        assert(len(message.tool_calls) <= 1) 
        return {"messages": [message]}
    
    # Initialize the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {"tools": "tools", END: END}
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def workflow(graph, user_input: str, thread_id: str = "1"):
    """Run workflow with true human interaction"""
    config = {"configurable": {"thread_id": thread_id}}
    input_message = {"messages": [{"role": "user", "content": user_input}]}
    
    for event in graph.stream(input_message,config,stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Check if we need human input
    snapshot = graph.get_state(config)
    
    if snapshot.next:  # If workflow is interrupted
        print(f"\n⏸️  Workflow paused. Waiting for human input...")
        
        # Get the interrupt data, e.g. {'query': 'I need some expert guidance for building an AI agent.'}
        interrupt_data = snapshot.tasks[0].interrupts[0].value
        
        # Get interactive human input
        human_response = get_input(interrupt_data)
        
        # Resume with human input
        resume_command = Command(resume=human_response)
        
        print(f"\n▶️  Resuming workflow with human input...\n")
        
        # Continue the workflow
        for event in graph.stream(resume_command, config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()
        print()

if __name__ == "__main__":
    """Main function to run the interactive demo"""
    
    # Create the graph
    graph = build_graph()
    
    # Example queries
    example_queries = [
        "Search for when Python programming language was first released, then use the human_assistance tool to verify the specific release date you found.",
        "Search for the release date of the first iPhone, then use human_assistance to verify the exact date you discovered.",
        "Look up for when LangGraph was released. When you have the answer, use the human_assistance tool for review."
    ]
    
    print("Choose a query or enter your own:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")
    
    choice = input(f"\nEnter choice (1-{len(example_queries)}) or type your own query: ").strip()
    user_query = example_queries[int(choice) - 1]
    
    # Run the workflow
    workflow(graph, user_query)