import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver 
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool

from dotenv import load_dotenv


# load env keys
load_dotenv()

# chat model
model=init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# Use free Hugging Face embeddings - better than google embedding model
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

# Tavily search tool
search = TavilySearch(max_results=2)


# manage memory: specify an ID for the memory thread
memory=MemorySaver()
config={"configurable": {"thread_id": "taido"}} # to pass in the generation call
agent=create_react_agent(
    model,
    tools=[retrieve, search],
    checkpointer=memory
)

# Save the graph visualization to a file
agent_png = agent.get_graph().draw_mermaid_png()
with open("agent_graph.png", "wb") as f:
    f.write(agent_png)
print("Graph visualization saved as 'agent_graph.png'")


if __name__=="__main__":
    print("Enter your question (or type 'exit' to quit)")
    while True:
        message=input(">> ")
        if message.lower()=='exit':
            break
        # form the input
        input_message={"messages": [{"role": "user", "content": message}]}
        # stream the output, use config to manage memory
        for event in agent.stream(
            input_message, 
            stream_mode="values",
            config=config
        ):
            event["messages"][-1].pretty_print()