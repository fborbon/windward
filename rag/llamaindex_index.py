"""Second, independent RAG stack — LlamaIndex over turbine spec/metadata (structured
facts: model, rated power, rotor diameter, coordinates, commercial-ops date), deliberately
separate from the LangChain/FAISS stack over incident narratives (rag/langchain_retriever.py).
Different framework, different corpus shape, same underlying Bedrock Titan embeddings.
"""
import json
from pathlib import Path
from typing import Any

import boto3
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from pydantic import PrivateAttr

import config
from agents.llm_router import complete as bedrock_complete
from data_sources.farms import FARMS, Farm

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "llamaindex" / "turbine_specs"
MODEL_ID = "amazon.titan-embed-text-v2:0"


class BedrockNovaLlamaLLM(CustomLLM):
    """Points LlamaIndex's response synthesis at the same Bedrock Nova model the LangGraph
    agent uses (via agents.llm_router), instead of defaulting to OpenAI."""

    context_window: int = 128000
    num_output: int = 512

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window, num_output=self.num_output, model_name="bedrock-nova-lite")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        response = bedrock_complete([{"role": "user", "content": prompt}])
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("streaming not needed for this project's LlamaIndex usage")

_index_cache = None


class BedrockTitanLlamaEmbedding(BaseEmbedding):
    """Reuses the same Titan v2 model as rag/embeddings.py, adapted to LlamaIndex's
    BaseEmbedding interface (a pydantic model — the boto3 client is a private attr,
    not a field, since it isn't pydantic-serializable)."""

    _client: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(model_name=MODEL_ID, **kwargs)
        self._client = boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)

    def _embed(self, text: str) -> list[float]:
        resp = self._client.invoke_model(modelId=MODEL_ID, body=json.dumps({"inputText": text[:8000]}))
        return json.loads(resp["body"].read())["embedding"]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)


def _turbine_spec_documents(farm: Farm) -> list[Document]:
    from data_sources.greenbyte_scada import load_turbine_static

    rows = load_turbine_static(farm.farm_id)

    docs = []
    for turbine_id, row in rows.iterrows():
        text = (
            f"{turbine_id}: {row.get('Manufacturer', '?')} {row.get('Model', '?')}, "
            f"rated power {row.get('Rated power (kW)', '?')} kW, hub height {row.get('Hub Height (m)', '?')} m, "
            f"rotor diameter {row.get('Rotor Diameter (m)', '?')} m, location "
            f"{row.get('Latitude', '?')}, {row.get('Longitude', '?')}, elevation {row.get('Elevation (m)', '?')} m, "
            f"commercial operations date {row.get('Commercial Operations Date', '?')}."
        )
        docs.append(Document(text=text, metadata={"source": f"turbine-spec:{turbine_id}"}))
    return docs


def build_index(farm_id: str) -> VectorStoreIndex:
    farm = FARMS[farm_id]
    path = INDEX_DIR / farm_id
    embed_model = BedrockTitanLlamaEmbedding()

    if (path / "docstore.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(path))
        return load_index_from_storage(storage_context, embed_model=embed_model)

    docs = _turbine_spec_documents(farm)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(path))
    return index


def query(farm_id: str, question: str) -> str:
    index = build_index(farm_id)
    engine = index.as_query_engine(llm=BedrockNovaLlamaLLM(), similarity_top_k=3)
    return str(engine.query(question))
