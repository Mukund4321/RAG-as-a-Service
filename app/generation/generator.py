from typing import List, Tuple
from app.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = (
    "You are a precise, citation-grounded assistant. "
    "Answer the question using ONLY the provided context passages. "
    "If the context does not contain enough information, say so clearly. "
    "Do not fabricate facts. Be concise and accurate."
)


def _build_context_block(chunks: List[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Source {i+1} — {chunk['filename']}, chunk {chunk['chunk_index']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, context_chunks: List[dict]) -> Tuple[str, int]:
    """
    Returns (answer_text, tokens_used).
    Selects backend based on settings.llm_backend.
    """
    if not context_chunks:
        return "No relevant documents found in the knowledge base.", 0

    context = _build_context_block(context_chunks)
    user_message = f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"

    if settings.llm_backend == "anthropic":
        return _generate_anthropic(user_message)
    else:
        return _generate_openai(user_message)


def _generate_anthropic(user_message: str) -> Tuple[str, int]:
    import anthropic
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text
    tokens = response.usage.input_tokens + response.usage.output_tokens
    return text, tokens


def _generate_openai(user_message: str) -> Tuple[str, int]:
    from openai import OpenAI
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    text = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return text, tokens
