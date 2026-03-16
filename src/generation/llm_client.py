from openai import AsyncOpenAI
from src.core.config import settings
from src.core.logger import logger


def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.vllm_url,
        api_key=settings.vllm_api_key,
    )


async def generate(prompt_messages: list, max_tokens: int = 512) -> str:
    """Call vLLM with messages and return the text response."""
    client = get_llm_client()
    try:
        response = await client.chat.completions.create(
            model=settings.vllm_model,
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("llm_error", error=str(e))
        raise
