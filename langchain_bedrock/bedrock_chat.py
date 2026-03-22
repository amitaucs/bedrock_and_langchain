import argparse
import os
from typing import Any
from uuid import uuid4

import boto3
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_aws import ChatBedrock
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID", "<KNOWLEDGE-BASE-ID>")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "<CLAUDE_INFERENCE_PROFILE_ID>")


def _response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts).strip()
    return str(content)


def _format_documents(documents: list[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in documents) or "No relevant context found."


def get_bedrock_client():
    """Initialize the Bedrock client using boto3 credential resolution."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
    )


def get_bedrock_agent_client():
    """Initialize the Bedrock agent runtime client for knowledge base retrieval."""
    return boto3.client(
        service_name="bedrock-agent-runtime",
        region_name=AWS_REGION,
    )


def create_retrieval_qa():
    """Create the retrieval-backed Bedrock chat system."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the provided context. "
                "If the context does not contain the answer, say that you do not "
                "have enough information.",
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    retriever = AmazonKnowledgeBasesRetriever(
        client=get_bedrock_agent_client(),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        region_name=AWS_REGION,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
    )

    llm = ChatBedrock(
        model_id=MODEL_ID,
        client=get_bedrock_client(),
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.0,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"],
        },
    )

    history_store: dict[str, BaseChatMessageHistory] = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in history_store:
            history_store[session_id] = InMemoryChatMessageHistory()
        return history_store[session_id]

    def run_qa(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs["question"]).strip()
        documents = retriever.invoke(question)
        response = llm.invoke(
            prompt.invoke(
                {
                    "history": inputs.get("history", []),
                    "context": _format_documents(documents),
                    "question": question,
                }
            )
        )
        return {
            "result": response,
            "source_documents": documents,
        }

    qa = RunnableWithMessageHistory(
        RunnableLambda(run_qa),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="result",
    )

    return qa, {"session_id": str(uuid4())}


def get_answer(query, qa, memory):
    """Get the answer for a query using the QA system."""
    result = qa.invoke(
        {"question": query},
         config={"configurable": {"session_id": memory["session_id"]}},
    )
    return _response_to_text(result["result"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Bedrock retrieval QA flow.")
    parser.add_argument(
        "-q",
        "--query",
        help="Single query to send to the knowledge base. If omitted, prompts interactively.",
    )
    args = parser.parse_args()

    qa, memory = create_retrieval_qa()

    try:
        query = args.query or input("Query: ").strip()
        if not query:
            print("No query provided.")
            return 1
        print(get_answer(query, qa, memory))
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
