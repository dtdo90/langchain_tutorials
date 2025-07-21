# LangChain Tutorials

This repository contains hands-on tutorials and example code for building advanced conversational AI workflows using [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph).

## Tutorials

1. **Chatbot basics**: Simple conversational chatbot with memory ([01_chatbot.py](01_chatbot.py))
2. **Retrieval-Augmented Generation (RAG)**: Search and answer using web data ([02_rag.py](02_rag.py))
3. **Agent workflows**: Multi-tool agent with search and retrieval ([03_agent.py](03_agent.py))
4. **Human-in-the-loop**: Interactive workflows with human verification 
([04_human_in_loop_basic.ipynb](04_human_in_loop_basic.ipynb), [04_human_in_loop_advanced.py](04_human_in_loop_advanced.py))

## Setup


#### Installation with uv (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd langchain_tutorials

# Install dependencies with uv
uv sync
```


### Environment Setup

1. Create a `.env` file in the project root:

2. Add your API keys to `.env`:
```env
# Google Gemini (required for most tutorials)
GOOGLE_API_KEY=your_google_api_key_here

# Tavily Search (required for agent and RAG tutorials)
TAVILY_API_KEY=your_tavily_api_key_here

```

