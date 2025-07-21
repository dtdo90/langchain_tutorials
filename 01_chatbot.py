from typing import Sequence

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, trim_messages
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

from typing_extensions import Annotated, TypedDict

# load enviroment API keys
load_dotenv()

# model
model=init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# define prompt template
prompt_template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all question to the best of your ability in {language}"
        ),
        MessagesPlaceholder(variable_name="messages"), # automatically insert system_prompt into memory messages
    ]
)


# message state for input: include messages (like MessageState) and language
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


trimmer=trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

# define action for graph node
def call_model(state: State):
    trimmed_messages=trimmer.invoke(state["messages"])
    prompt=prompt_template.invoke({"messages": trimmed_messages, "language": state["language"]})
    response=model.invoke(prompt)
    return {"message": [response]}


# Define graph
workflow=StateGraph(state_schema=State)
workflow.add_edge(START, "model") # set "model" as entry point
workflow.add_node("model", call_model) # set "model" as a node with action call_model
memory=MemorySaver()

# compile the graph
app=workflow.compile(checkpointer=memory)
print("Graph compiled successfully!")

# define config for memory
config={"configurable": {"thread_id": "taido"}}

if __name__=="__main__":
    print("Enter your language (or 'exit' to quit)")
    language=input(">> ")
    print("Enter your query (or 'exit' to quit)")
    while True:
        query=input(">> ")
        if query.lower()=="exit":
            break
        input_messages=[HumanMessage(query)]
        # streaming
        for chunk, metadata in app.stream(
            {"messages": input_messages, "language": language},
            config,
            stream_mode="messages"
        ):
            if isinstance(chunk, AIMessage): 
                print(chunk.content, end="")
