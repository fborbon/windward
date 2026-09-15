"""Langfuse tracing for the LangGraph agent — free tier on cloud.langfuse.com, or
self-hostable. Auto-activates when LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are set;
otherwise every graph run just proceeds untraced.

Langfuse v4's LangChain/LangGraph integration reads credentials from the environment
(not constructor args) — see https://langfuse.com/integrations/frameworks/langgraph.
"""
import os

import config


def get_handler():
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return None

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)

    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


def wrap_completion(fn):
    """Wraps agents.llm_router's raw litellm.completion() call with a manual Langfuse
    generation span, for call sites that don't run through a LangChain Runnable (get_handler()
    above only covers LangGraph/LangChain via config={"callbacks": [...]}, which
    agents/qa_agent.py's tool-calling loop doesn't use).

    Deliberately NOT litellm's own built-in success_callback=["langfuse"] hook: as of
    litellm 1.100-1.101 + langfuse 4.15, that path is broken — it does
    `langfuse.version.__version__`, which doesn't exist on the v4 SDK
    (AttributeError: module 'langfuse' has no attribute 'version'), and litellm swallows the
    error as "non-blocking" but it still aborts the actual completion() call, silently
    degrading agents/qa_agent.py's tool-calling loop to its stuffed-prompt fallback on every
    call. Manual instrumentation via langfuse.get_client().start_as_current_observation()
    sidesteps that broken code path entirely. Caller must only apply this when
    LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are set (see agents/llm_router.py) — it's not
    itself a no-op."""
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)

    from langfuse import get_client

    def wrapped(messages, **kwargs):
        client = get_client()
        with client.start_as_current_observation(
            as_type="generation", name="windward-complete", model=kwargs.get("model", "bedrock"), input=messages
        ) as gen:
            try:
                response = fn(messages, **kwargs)
            except Exception as e:
                gen.update(level="ERROR", status_message=str(e))
                raise
            usage = getattr(response, "usage", None)
            gen.update(
                output=response.choices[0].message.content,
                usage_details=(
                    {"input": usage.prompt_tokens, "output": usage.completion_tokens} if usage else None
                ),
            )
            return response

    return wrapped
