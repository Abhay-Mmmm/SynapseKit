"""
SynapseKit — lightweight, async-first RAG framework.

3-line happy path:

    from synapsekit import RAG

    rag = RAG(model="gpt-4o-mini", api_key="sk-...")
    rag.add("Your document text here")

    async for token in rag.stream("What is the main topic?"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

from .agents import (
    AgentConfig,
    AgentExecutor,
    AgentMemory,
    AgentStep,
    BaseTool,
    CalculatorTool,
    FileReadTool,
    FunctionCallingAgent,
    PythonREPLTool,
    ReActAgent,
    SQLQueryTool,
    ToolRegistry,
    ToolResult,
    WebSearchTool,
    tool,
)
from .embeddings.backend import SynapsekitEmbeddings
from .graph import (
    END,
    BaseCheckpointer,
    CompiledGraph,
    ConditionalEdge,
    ConditionFn,
    Edge,
    GraphConfigError,
    GraphRuntimeError,
    GraphState,
    InMemoryCheckpointer,
    Node,
    NodeFn,
    SQLiteCheckpointer,
    StateGraph,
    agent_node,
    rag_node,
)
from .llm.base import BaseLLM, LLMConfig
from .loaders.base import Document
from .loaders.csv import CSVLoader
from .loaders.directory import DirectoryLoader
from .loaders.html import HTMLLoader
from .loaders.json_loader import JSONLoader
from .loaders.pdf import PDFLoader
from .loaders.text import StringLoader, TextLoader
from .loaders.web import WebLoader
from .memory.conversation import ConversationMemory
from .observability.tracer import TokenTracer
from .parsers.json_parser import JSONParser
from .parsers.list_parser import ListParser
from .parsers.pydantic_parser import PydanticParser
from .prompts.template import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from .rag.facade import RAG
from .rag.pipeline import RAGConfig, RAGPipeline
from .retrieval.base import VectorStore
from .retrieval.retriever import Retriever
from .retrieval.vectorstore import InMemoryVectorStore
from .text_splitters import (
    BaseSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticSplitter,
    TokenAwareSplitter,
)

__version__ = "0.5.1"
__all__ = [
    # Facade
    "RAG",
    # Pipeline
    "RAGPipeline",
    "RAGConfig",
    # LLM
    "BaseLLM",
    "LLMConfig",
    # Embeddings
    "SynapsekitEmbeddings",
    # Vector stores
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    # Retrieval
    "Retriever",
    # Memory / observability
    "ConversationMemory",
    "TokenTracer",
    # Loaders
    "Document",
    "TextLoader",
    "StringLoader",
    "PDFLoader",
    "HTMLLoader",
    "CSVLoader",
    "JSONLoader",
    "DirectoryLoader",
    "WebLoader",
    # Parsers
    "JSONParser",
    "PydanticParser",
    "ListParser",
    # Prompts
    "PromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    # Agents
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "AgentMemory",
    "AgentStep",
    "ReActAgent",
    "FunctionCallingAgent",
    "AgentExecutor",
    "AgentConfig",
    # Tool decorator
    "tool",
    # Built-in tools
    "CalculatorTool",
    "FileReadTool",
    "PythonREPLTool",
    "SQLQueryTool",
    "WebSearchTool",
    # Text splitters
    "BaseSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenAwareSplitter",
    "SemanticSplitter",
    # Graph workflows
    "END",
    "GraphState",
    "GraphConfigError",
    "GraphRuntimeError",
    "Node",
    "NodeFn",
    "agent_node",
    "rag_node",
    "Edge",
    "ConditionalEdge",
    "ConditionFn",
    "StateGraph",
    "CompiledGraph",
    # Checkpointers
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "SQLiteCheckpointer",
]

# Lazy imports for optional vector store backends
_OPTIONAL_VECTOR_STORES = {
    "ChromaVectorStore": "retrieval.chroma",
    "FAISSVectorStore": "retrieval.faiss",
    "QdrantVectorStore": "retrieval.qdrant",
    "PineconeVectorStore": "retrieval.pinecone",
}


def __getattr__(name: str):
    if name in _OPTIONAL_VECTOR_STORES:
        import importlib

        mod = importlib.import_module(f".{_OPTIONAL_VECTOR_STORES[name]}", __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
