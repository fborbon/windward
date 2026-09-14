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


def enable_litellm_tracing():
    """Enables litellm's own native Langfuse callback for raw complete()/litellm.completion()
    calls that don't run through a LangChain Runnable — get_handler() above only covers
    LangGraph/LangChain via the config={"callbacks": [...]} kwarg, which agents/qa_agent.py's
    tool-calling loop doesn't use. No-op under the same conditions as get_handler()."""
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)

    import litellm

    if "langfuse" not in litellm.success_callback:
        litellm.success_callback.append("langfuse")
    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback.append("langfuse")
