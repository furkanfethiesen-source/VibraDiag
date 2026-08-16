
import yaml
from typing import Dict, Any
from functools import lru_cache
from pydantic import BaseModel


class AppConfig(BaseModel):
    app: Dict[str, Any]
    llm: Dict[str, Any]
    embeddings: Dict[str, Any]
    streamlit: Dict[str, Any]
    logging: Dict[str, Any]
    vision_llm: Dict[str, Any]
    paths: Dict[str, Any] = {}
    decomposer: Dict[str, Any] = {}
    self_corrector: Dict[str, Any] = {}

class RetrievalConfig(BaseModel):
    parent_child_retrieval: Dict[str, Any] = {}
    hybrid_search: Dict[str, Any] = {}
    chromadb: Dict[str, Any] = {}
    parser: Dict[str, Any] = {}
    visual_retrieval: Dict[str, Any] = {}
    qdrant: Dict[str, Any] = {}
    docstore: Dict[str, Any] = {}

class PromptsConfig(BaseModel):
    diagnosis_prompt: Dict[str, Any] = {}
    explanation_prompt: Dict[str, Any] = {}
    parsing_prompts: Dict[str, Any] = {}
    generation_prompt: Dict[str, Any] = {}
    decomposer_prompt: Dict[str, Any] = {}
    corrector_prompts: Dict[str, Any] = {}


@lru_cache(maxsize=1)
def load_appcfg(config_path: str = "config/app.yaml") -> AppConfig:
    with open(config_path, 'r') as f:
        return AppConfig(**yaml.safe_load(f))

@lru_cache(maxsize=1)
def load_retcfg(config_path: str = "config/retrieval.yaml") -> RetrievalConfig:
    with open(config_path, 'r') as f:
        return RetrievalConfig(**yaml.safe_load(f))

@lru_cache(maxsize=1)
def load_prompts_cfg(config_path: str = "config/prompts.yaml") -> PromptsConfig:
    with open(config_path, 'r') as f:
        return PromptsConfig(**yaml.safe_load(f))



