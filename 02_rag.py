import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver 

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

from langchain_core.messages import SystemMessage, trim_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END
from dotenv import load_dotenv

# load env keys
load_dotenv()

# chat model
model=init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# Use free Hugging Face embeddings - excellent performance and completely free
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# vector store to manage embedded vectors
vector_store=InMemoryVectorStore(embeddings)

# load and chunk contents of the blog
loader=WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    )
)
docs=loader.load()

# Use standard chunking parameters
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits=text_splitter.split_documents(docs)

# index chunks into vector store
_ = vector_store.add_documents(documents=all_splits)

# define retrieve step as a tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs=vector_store.similarity_search(query,k=2)
    serialized="\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# 1. Generate an AI message that may include a tool-call to be sent (could be retrieval message)
def query_or_response(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    llm_with_tools=model.bind_tools([retrieve])
    # llm will automaticall generate a "query" for search. This query is better than the original query
    response=llm_with_tools.invoke(state["messages"])
    # MessageState appends messages to state instead of overwritting
    return {"messages": [response]}

# 2. Execute the retrieval: add this as a node in the graph
tools=ToolNode([retrieve]) # retrieve is the node's action

# 3. Generate a response using the retrieved content
# trimmer to restrict conversation history
trimmer=trim_messages(
    max_tokens=4000, # set larger for more memory
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages: those that are retrieved by RAG
    recent_tool_messages=[]
    for message in reversed(state["messages"]): # state with the most recent one
        if message.type=="tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages=recent_tool_messages[::-1] # reverse the list
    
    # format prompt
    docs_content="\n\n".join(doc.content for doc in tool_messages)
    system_message=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        f"{docs_content}"
    )
    conversation_messages=[
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type=="ai" and not message.tool_calls)
    ]
    prompt=[SystemMessage(system_message)]+conversation_messages
    # trim the message so that it doesn't get too long
    trimmed_prompt=trimmer.invoke(prompt)
    response=model.invoke(trimmed_prompt)
    return {"messages": [response]}


# Build graph
graph_builder=StateGraph(MessagesState)
graph_builder.add_node(query_or_response)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_response")
graph_builder.add_conditional_edges(
    "query_or_response",
    tools_condition, # use tool or not
    {END:END, "tools": "tools"} # END node or "tools" node
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

# manage memory: specify an ID for the memory thread
memory=MemorySaver()
config={"configurable": {"thread_id": "taido"}}

# compile the graph
graph=graph_builder.compile(checkpointer=memory) # specify memory checkpointer

# Save the graph visualization to a file
graph_png = graph.get_graph().draw_mermaid_png()
with open("rag_graph.png", "wb") as f:
    f.write(graph_png)
print("Graph visualization saved as 'rag_graph.png'")


if __name__=="__main__":
    print("Enter your question (or type 'exit' to quit)")
    while True:
        message=input(">> ")
        if message.lower()=='exit':
            break
        # form the input
        input_message={"messages": [{"role": "user", "content":message}]}
        # stream the output, use config to manage memory
        for step in graph.stream(
            input_message, 
            stream_mode="values",
            config=config
        ):
            step["messages"][-1].pretty_print()
