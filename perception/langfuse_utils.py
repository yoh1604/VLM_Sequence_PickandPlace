import os
from contextlib import contextmanager

try:
    from langfuse import get_client
except Exception:
    get_client = None


def is_langfuse_enabled():
    return (
        os.getenv("LANGFUSE_ENABLED", "0") == "1"
        and get_client is not None
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def get_langfuse():
    if not is_langfuse_enabled():
        return None
    return get_client()


def flush_langfuse():
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception as e:
            print(f"[LANGFUSE] flush failed: {e}")


def ollama_usage_details(response):
    """
    Mapping usage Ollama -> Langfuse.
    Ollama biasanya mengembalikan:
    - prompt_eval_count = input tokens
    - eval_count = output tokens
    """
    if not isinstance(response, dict):
        return {}

    input_tokens = int(response.get("prompt_eval_count", 0) or 0)
    output_tokens = int(response.get("eval_count", 0) or 0)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }


@contextmanager
def trace_generation(name, model, input_data=None, metadata=None):
    lf = get_langfuse()

    if lf is None:
        yield None
        return

    try:
        with lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_data,
            metadata=metadata or {},
        ) as generation:
            yield generation
    except Exception as e:
        print(f"[LANGFUSE] tracing skipped: {e}")
        yield None
