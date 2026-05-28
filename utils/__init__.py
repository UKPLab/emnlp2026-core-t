"""Shared utilities package for CORE-T."""

from .utils import (
    DatabaseType,
    Configuration,
    get_results_run_dir,
    make_run_tag,
    create_llm,
    create_embeddings,
    create_reranker,
    llm_usage_from_prompt_and_output,
    summarize_llm_invoke_error,
)
from .cache_scripts.faiss_vector_cache import FAISSVectorCache, get_faiss_vector_cache
from .cache_scripts.metadata_cache import MetadataCache, get_metadata_cache
from .cache_scripts.sql_generation_cache import (
    SQLExecutionCache,
    SQLGenerationCache,
    get_sql_execution_cache,
    get_sql_generation_cache,
)
from .cache_scripts.selection_cache import SelectionCache, get_selection_cache

# Re-export common helpers so callers can `from utils import ...`
__all__ = [
    "DatabaseType",
    "Configuration",
    "make_run_tag",
    "get_results_run_dir",
    "create_llm",
    "create_embeddings",
    "create_reranker",
    "llm_usage_from_prompt_and_output",
    "summarize_llm_invoke_error",
    "MetadataCache",
    "get_metadata_cache",
    "SelectionCache",
    "get_selection_cache",
    "SQLExecutionCache",
    "SQLGenerationCache",
    "get_sql_execution_cache",
    "get_sql_generation_cache",
    "FAISSVectorCache",
    "get_faiss_vector_cache",
]
