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


def _first_content_heading(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("#").strip()
        if 4 <= len(line) <= 100:
            return line
    return None


def _as_clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_as_clean_text(item) for item in value]
        joined = ", ".join(part for part in parts if part)
        return joined or None
    return None


def _extract_source_label(metadata: dict[str, Any]) -> str | None:
    direct_keys = (
        "x-amz-bedrock-kb-source-uri",
        "source",
        "uri",
        "url",
        "file_path",
        "filename",
        "name",
        "title",
    )
    for key in direct_keys:
        value = _as_clean_text(metadata.get(key))
        if value:
            return value

    location = metadata.get("location")
    if isinstance(location, dict):
        for nested_key in ("uri", "url", "path"):
            value = _as_clean_text(location.get(nested_key))
            if value:
                return value
        for nested_key in (
            "s3Location",
            "webLocation",
            "confluenceLocation",
            "salesforceLocation",
            "sharePointLocation",
            "customDocumentLocation",
        ):
            nested = location.get(nested_key)
            if isinstance(nested, dict):
                for field in ("uri", "url", "path"):
                    value = _as_clean_text(nested.get(field))
                    if value:
                        return value

    return None


def _extract_section_reference(document: Any) -> str | None:
    metadata = getattr(document, "metadata", {}) or {}

    section_like_keys = (
        "section",
        "section_title",
        "section_name",
        "heading",
        "headings",
        "header",
        "subsection",
        "chapter",
        "topic",
        "page",
        "page_number",
    )
    for key in section_like_keys:
        value = _as_clean_text(metadata.get(key))
        if value:
            label = "Page" if key in {"page", "page_number"} else "Section"
            return f"{label}: {value}"

    document_attributes = metadata.get("document_attributes")
    if isinstance(document_attributes, dict):
        for key in section_like_keys:
            value = _as_clean_text(document_attributes.get(key))
            if value:
                label = "Page" if key in {"page", "page_number"} else "Section"
                return f"{label}: {value}"

    content_heading = _first_content_heading(getattr(document, "page_content", "") or "")
    if content_heading:
        return f"Excerpt: {content_heading}"

    return None


def _format_source_references(documents: list[Any]) -> str:
    references: list[str] = []
    seen: set[tuple[str, str]] = set()

    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source_label = _extract_source_label(metadata) or "Unknown source"
        section_reference = _extract_section_reference(document) or "Section unavailable"
        dedupe_key = (source_label, section_reference)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        references.append(f"- {source_label} | {section_reference}")

    if not references:
        return ""

    return "\n".join(["Source References:"] + references)


def _format_answer_with_sources(response: Any, documents: list[Any]) -> str:
    answer_text = _response_to_text(response)
    source_references = _format_source_references(documents)
    if not source_references:
        return answer_text
    return f"{answer_text}\n\n{source_references}".strip()


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
            "formatted_answer": _format_answer_with_sources(response, documents),
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
    return result.get("formatted_answer") or _format_answer_with_sources(
        result["result"],
        result.get("source_documents", []),
    )


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
