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
