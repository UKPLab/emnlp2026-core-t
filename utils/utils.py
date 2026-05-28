"""Lightweight helpers shared by CORE-T scripts."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _truncate_debug_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def summarize_llm_invoke_error(exc: BaseException, *, max_len: int = 4000) -> str:
    """Collect exception text plus HTTP/SDK payloads (OpenRouter often returns a generic top-level message)."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    for _ in range(10):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        main = str(cur).strip() or cur.__class__.__name__
        parts.append(main)

        resp = getattr(cur, "response", None)
        if resp is not None:
            try:
                text = getattr(resp, "text", "") or ""
                if text.strip():
                    parts.append(f"http_response={_truncate_debug_text(text, 1800)}")
            except Exception:
                pass
        try:
            body = getattr(cur, "body", None)
            if body is not None and str(body).strip():
                parts.append(f"sdk_body={_truncate_debug_text(str(body), 1800)}")
        except Exception:
            pass

        cur = getattr(cur, "__cause__", None)

    joined = " | ".join(parts)
    return joined[:max_len]


def _sanitize_path_component(value: object) -> str:
    """Best-effort filesystem-safe-ish component for cache/results paths."""
    s = str(value or "").strip()
    if not s:
        return "unknown"
    return "".join(c if c.isalnum() else "_" for c in s)


def make_run_tag(*, llm_model: str, embedding_model: str) -> str:
    """Return `{llm}_{embedding_model}` with safe path characters."""
    return f"{_sanitize_path_component(llm_model)}_{_sanitize_path_component(embedding_model)}"


def get_results_run_dir(
    *,
    dataset: str,
    step_dirname: str,
    llm_model: str,
    embedding_model: str,
    project_root: Optional[Path] = None,
) -> Path:
    """Return `<project_root>/results/<dataset>/{llm}_{embedding_model}/<step_dirname>`."""
    root = project_root or _detect_project_root()
    run_tag = make_run_tag(llm_model=llm_model, embedding_model=embedding_model)
    return (root / "results" / str(dataset) / run_tag / str(step_dirname)).resolve()


class DatabaseType(str, Enum):
    """Supported database types."""

    SPIDER = "spider"
    BIRD = "bird"
    BEAVER = "beaver"
    MMQA = "mmqa"


class Configuration:
    """Minimal configuration used by CORE-T scripts."""

    def __init__(
        self,
        database_type: DatabaseType = DatabaseType.BIRD,
        llm_model: str = "openai:gpt-4o-mini",
        embedding_model: str = "fireworks:WhereIsAI/UAE-Large-V1",
        temperature: float = 0.0,
        **_kwargs,
    ):
        self.database_type = database_type
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature

    def get_database_cache_dir(self, name: str) -> str:
        """Return `<project_root>/cache/<dataset>/{llm}_{embedding_model}/<name>`."""
        root = _detect_project_root()
        run_tag = make_run_tag(llm_model=self.llm_model, embedding_model=self.embedding_model)
        return str(root / "cache" / self.database_type.value / run_tag / name)

    def get_dataset_cache_dir(self, name: str) -> str:
        """Return `<project_root>/cache/<dataset>/<name>` (no LLM/embedding run tag).

        Use for caches shared across table-selection / embedding configurations.
        """
        root = _detect_project_root()
        return str(root / "cache" / self.database_type.value / name)

    def create_llm(self):
        """Create a chat model used by standalone scripts."""
        return create_llm(self.llm_model, temperature=self.temperature)

    def format_prompt(self, prompt: str) -> str:
        """Hook to adapt prompts if needed. For now, keep as plain text."""
        return prompt


def _detect_project_root() -> Path:
    """Best-effort project root detection (looks for `data/` or `sql_database/`)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").exists() or (parent / "sql_database").exists():
            return parent
    return here.parent.parent


def configure_hf_hub_cache(
    project_root: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """Point Hugging Face caches at ``<project_root>/.hf_cache``.

    Compute nodes often lack a writable ``~/.cache/huggingface``. On SLURM jobs we
    always redirect caches to project storage (override inherited login-node paths).
    """
    root = project_root or _detect_project_root()
    hf_home = root / ".hf_cache"
    transformers_cache = hf_home / "transformers"
    hub_cache = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    on_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    env_map = {
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(transformers_cache),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
    }
    for key, value in env_map.items():
        if force or on_slurm or key not in os.environ:
            os.environ[key] = value
    return hf_home


def extract_tokenizer(model: Any) -> Optional[Any]:
    """Best-effort tokenizer lookup across LangChain/HF wrappers."""
    candidates = [
        ("llm", "pipeline", "tokenizer"),
        ("pipeline", "tokenizer"),
        ("tokenizer",),
    ]
    for chain in candidates:
        obj = model
        ok = True
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            return obj
    return None


def estimate_token_count(text: str, tokenizer: Optional[Any]) -> Optional[int]:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(str(text), add_special_tokens=False)
            if isinstance(ids, list):
                return int(len(ids))
        except Exception:
            pass
        try:
            out = tokenizer(str(text), add_special_tokens=False)
            if isinstance(out, dict) and isinstance(out.get("input_ids"), list):
                return int(len(out.get("input_ids")))
        except Exception:
            pass
    try:
        return int(max(1, len(str(text).split())))
    except Exception:
        return None


def llm_usage_from_prompt_and_output(
    model: Any,
    prompt_text: str,
    output_text: str,
    *,
    ai_message: Optional[Any] = None,
    cost_usd: float = 0.0,
) -> Dict[str, Any]:
    """
    Provider token usage from message metadata when present; else tokenizer estimates.

    Returns a dict suitable for cluster_summary.llm_usage / eval aggregation:
    input_tokens, output_tokens, total_tokens, cost_usd
    """
    llm_usage: Dict[str, Any] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    if ai_message is not None:
        try:
            meta = getattr(ai_message, "response_metadata", {}) or {}
            usage_meta = meta.get("token_usage") or getattr(ai_message, "usage_metadata", {}) or {}
            llm_usage["input_tokens"] = usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens")
            llm_usage["output_tokens"] = usage_meta.get("output_tokens") or usage_meta.get("completion_tokens")
            llm_usage["total_tokens"] = usage_meta.get("total_tokens")
        except Exception:
            pass

    try:
        if (
            llm_usage.get("input_tokens") is None
            and llm_usage.get("output_tokens") is None
            and llm_usage.get("total_tokens") is None
        ):
            tokenizer = extract_tokenizer(model)
            in_tok = estimate_token_count(prompt_text, tokenizer)
            out_tok = estimate_token_count(output_text, tokenizer)
            if in_tok is not None:
                llm_usage["input_tokens"] = int(in_tok)
            if out_tok is not None:
                llm_usage["output_tokens"] = int(out_tok)
            if in_tok is not None and out_tok is not None:
                llm_usage["total_tokens"] = int(in_tok + out_tok)
    except Exception:
        pass

    out = dict(llm_usage)
    out["cost_usd"] = float(cost_usd)
    return out


def _is_huggingface_model(model_name: str) -> bool:
    return isinstance(model_name, str) and (
        model_name.startswith("huggingface:") or model_name.startswith("hf:")
    )


_init_chat_model_fn: Optional[Callable[..., Any]] = None


def _get_init_chat_model() -> Callable[..., Any]:
    """Resolve LangChain's `init_chat_model` once (first use), not on every `create_llm` call."""
    global _init_chat_model_fn
    if _init_chat_model_fn is None:
        try:
            from langchain.chat_models import init_chat_model as _fn
        except ImportError:  # pragma: no cover
            from langchain.chat_models.base import init_chat_model as _fn

        _init_chat_model_fn = _fn
    return _init_chat_model_fn


_init_embeddings_fn: Optional[Callable[..., Any]] = None


def _get_init_embeddings() -> Callable[..., Any]:
    """Resolve LangChain's `init_embeddings` once (first use), not on every `create_embeddings` call."""
    global _init_embeddings_fn
    if _init_embeddings_fn is None:
        try:
            from langchain.embeddings import init_embeddings as _fn
        except ImportError:  # pragma: no cover
            from langchain.embeddings.base import init_embeddings as _fn

        _init_embeddings_fn = _fn
    return _init_embeddings_fn


def _create_openrouter_chat(model_name: str, *, temperature: float) -> Any:
    """OpenRouter via `langchain-openrouter` (more reliable than `init_chat_model` for this provider)."""
    raw_model = model_name.split(":", 1)[1].strip() if ":" in model_name else ""
    if not raw_model:
        raise ValueError(
            "OpenRouter model id is required, e.g. openrouter:meta-llama/llama-3.2-3b-instruct"
        )
    try:
        from langchain_openrouter import ChatOpenRouter
    except Exception as import_exc:  # pragma: no cover
        raise ImportError(
            "OpenRouter support requires langchain-openrouter. "
            "Install with: pip install -U langchain-openrouter"
        ) from import_exc

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
    # Keep completion budget modest: large max_tokens + long prompts can exceed
    # provider context and surface as generic OpenRouter "Provider returned error".
    kwargs: Dict[str, Any] = {
        "model": raw_model,
        "temperature": float(temperature or 0.0),
        "max_tokens": 8192,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenRouter(**kwargs)


def _create_huggingface_chat(model_name: str, *, temperature: float = 0.0):
    """Create a ChatHuggingFace model backed by a transformers pipeline.

    This mirrors the cluster setup used for:
    - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen2.5-7B-Instruct
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  # type: ignore
    except Exception as import_exc:  # pragma: no cover
        raise ImportError(
            "Hugging Face support requires 'transformers', 'torch', and 'langchain-huggingface'. "
            "Install with: pip install -U transformers torch langchain-huggingface"
        ) from import_exc

    # Strip provider prefix to get the HF repo id.
    repo_id = model_name.split(":", 1)[1] if ":" in model_name else model_name

    is_llama_3_1_8b = repo_id == "meta-llama/Llama-3.1-8B-Instruct"
    is_qwen_2_5_7b = repo_id == "Qwen/Qwen2.5-7B-Instruct"

    configure_hf_hub_cache()

    # Respect an HF token from environment for gated repos.
    try:
        hf_token = (
            os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True,
        cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
    )
    if is_llama_3_1_8b and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use device_map="auto" only if accelerate is installed.
    has_accelerate = False
    try:
        import accelerate  # type: ignore  # noqa: F401

        has_accelerate = True
    except Exception:
        has_accelerate = False

    try:
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if (is_llama_3_1_8b or is_qwen_2_5_7b) else "auto",
            "cache_dir": os.environ.get("TRANSFORMERS_CACHE"),
        }
        if has_accelerate:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            cache_dir=os.environ.get("TRANSFORMERS_CACHE"),
        )

    do_sample = bool(temperature and float(temperature) > 0.0)
    gen_kwargs = {
        "max_new_tokens": 16384,
        "do_sample": do_sample,
        "pad_token_id": getattr(tokenizer, "eos_token_id", None),
        "return_full_text": False,
    }
    if do_sample:
        gen_kwargs["temperature"] = max(0.0, float(temperature or 0.0))

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **gen_kwargs,
    )
    # Explicitly clear wrapper-level generation kwargs. Some langchain-huggingface
    # versions otherwise inject sampling params (temperature/top_p/top_k) that are
    # not accepted by every transformers backend and emit noisy warnings.
    try:
        hf_llm = HuggingFacePipeline(pipeline=gen_pipe, model_kwargs={}, pipeline_kwargs={})
    except TypeError:
        hf_llm = HuggingFacePipeline(pipeline=gen_pipe)

    # For these instruct models, apply the tokenizer chat template.
    if is_llama_3_1_8b or is_qwen_2_5_7b:

        def _convert_messages_to_prompt(messages):
            chat = []
            for m in messages:
                m_type = getattr(m, "type", "human")
                if m_type == "system":
                    role = "system"
                elif m_type == "human":
                    role = "user"
                elif m_type in ("ai", "assistant"):
                    role = "assistant"
                else:
                    role = "user"
                chat.append({"role": role, "content": getattr(m, "content", "")})
            return tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )

        return ChatHuggingFace(llm=hf_llm, convert_messages_to_prompt=_convert_messages_to_prompt)

    return ChatHuggingFace(llm=hf_llm)


def create_llm(model_name: str, *, temperature: float = 0.0):
    """Create a chat model.

    Supports common LangChain providers (OpenAI, Together, Gemini, etc.) via
    `init_chat_model`, and additionally supports local Hugging Face models via:
    - `huggingface:<repo_id>`
    - `hf:<repo_id>`
    """
    if not model_name:
        raise ValueError("model_name is required")

    model_str = str(model_name).strip()
    temp = float(temperature or 0.0)

    # Hugging Face (local) path first: `init_chat_model` doesn't handle this prefix.
    if _is_huggingface_model(model_name):
        return _create_huggingface_chat(model_str, temperature=temp)

    # OpenRouter: use the dedicated integration (init_chat_model often fails for this provider).
    if model_str.lower().startswith("openrouter:"):
        return _create_openrouter_chat(model_str, temperature=temp)

    # DeepInfra is not (always) registered as an `init_chat_model` provider. Use the
    # community integration directly.
    if model_str.lower().startswith("deepinfra:"):
        raw_model = model_str.split(":", 1)[1].strip()
        if not raw_model:
            raise ValueError("DeepInfra model is required, e.g. deepinfra:google/gemma-3-4b-it")
        try:
            from langchain_community.chat_models import ChatDeepInfra  # type: ignore
        except Exception as import_exc:  # pragma: no cover
            raise ImportError(
                "DeepInfra chat support requires langchain-community to be installed."
            ) from import_exc

        api_token = (
            os.environ.get("DEEPINFRA_API_TOKEN")
            or os.environ.get("DEEPINFRA_API_KEY")
            or os.environ.get("DEEPINFRA_TOKEN")
        )
        # ChatDeepInfra can still pick up env vars internally; we pass it if present.
        return ChatDeepInfra(
            model=raw_model,
            temperature=temp,
            max_tokens=16384,
            deepinfra_api_token=api_token,
        )

    init_chat_model = _get_init_chat_model()

    if ":" in model_str:
        provider, raw_model = (s.strip() for s in model_str.split(":", 1))
        if provider and raw_model:
            try:
                return init_chat_model(raw_model, model_provider=provider, temperature=temp)
            except TypeError:
                # Older/newer LangChain variants or wrappers; fall back.
                pass
            except Exception as first_exc:
                # Preserve backward compatibility: if the original `provider:model`
                # string happens to work for this provider in this environment, use it.
                try:
                    return init_chat_model(model_str, temperature=temp)
                except Exception:
                    raise first_exc

    return init_chat_model(model_str, temperature=temp)


def create_embeddings(model_name: str):
    """Create embeddings using LangChain initializers."""

    configure_hf_hub_cache()

    if not model_name:
        raise ValueError("embedding model name is required")

    lower = model_name.lower()
    snowflake_aliases = {
        "snowflake",
        "arctic",
        "arctic-embed",
        "snowflake-arctic-embed-m-v2",
        "snowflake-arctic-embed-m-v2.0",
        "snowflake/snowflake-arctic-embed-m-v2.0",
    }
    if (
        lower in snowflake_aliases
        or lower.startswith("snowflake:")
        or lower.startswith("snowflake/")
    ):
        normalized = (
            "Snowflake/snowflake-arctic-embed-m-v2.0"
            if lower in snowflake_aliases
            else model_name.split(":", 1)[1]
            if lower.startswith("snowflake:")
            else model_name
        )
        return _create_snowflake_embeddings(normalized)

    selected_model = model_name
    if ":" not in selected_model:
        selected_model = f"openai:{selected_model}"

    if selected_model.startswith("fireworks:"):
        fireworks_model = selected_model.split(":", 1)[1]
        from langchain_fireworks import FireworksEmbeddings

        return FireworksEmbeddings(model=fireworks_model)

    return _get_init_embeddings()(selected_model)


def _create_snowflake_embeddings(model_name: str):
    """Snowflake Arctic embeddings adapter."""
    try:
        import torch
    except Exception:
        torch = None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required for Snowflake embeddings. Install with: pip install sentence-transformers"
        ) from e

    has_cuda = bool(
        torch is not None
        and hasattr(torch, "cuda")
        and callable(getattr(torch.cuda, "is_available", None))
        and torch.cuda.is_available()
    )
    device = "cuda" if has_cuda else "cpu"

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        config_kwargs={"use_memory_efficient_attention": False},
    )
    model = model.to(device)

    class SnowflakeEmbeddingsAdapter:
        """Minimal adapter exposing LangChain's embeddings interface."""

        def __init__(self, _model):
            self._model = _model

        def embed_documents(self, texts):
            if not texts:
                return []
            vecs = self._model.encode(
                texts,
                prompt_name="document",
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vecs.tolist()

        def embed_query(self, text):
            if not text:
                return []
            vec = self._model.encode(
                [text],
                prompt_name="query",
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return vec[0].tolist()

    return SnowflakeEmbeddingsAdapter(model)


def create_reranker(model_name: str):
    """Create a reranker adapter for Together, Cohere, or Fireworks."""
    if not model_name:
        raise ValueError("reranker_model is required when reranking is enabled")

    if model_name.startswith("fireworks:"):
        fw_model = model_name.split(":", 1)[1].strip()
        import requests

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("Missing FIREWORKS_API_KEY env var for Fireworks reranking")

        url = "https://api.fireworks.ai/inference/v1/rerank"

        class FireworksRerankerAdapter:
            def __init__(self, model_str: str, api_key: str):
                self._model = model_str
                self._api_key = api_key

            def rerank(self, *, query: str, documents: list, top_n: int | None = None):
                payload = {
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "return_documents": False,
                }
                if top_n is not None:
                    payload["top_n"] = int(top_n)

                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }

                r = requests.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.json()

                out = []
                for item in data.get("data", []):
                    out.append(
                        {
                            "index": int(item["index"]),
                            "score": float(item.get("relevance_score", 0.0)),
                        }
                    )
                out.sort(key=lambda x: x["score"], reverse=True)
                if top_n is not None:
                    out = out[: int(top_n)]
                return out

        return FireworksRerankerAdapter(fw_model, api_key)

    if model_name.startswith("cohere:"):
        cohere_model = model_name.split(":", 1)[1]
        from cohere import Client 

        client = Client()  # picks up COHERE_API_KEY

        class CohereRerankerAdapter:
            def __init__(self, client_inst, model_str):
                self._client = client_inst
                self._model = model_str

            def rerank(self, *, query: str, documents: list, top_n: int | None = None):
                resp = self._client.rerank(
                    model=self._model,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                )
                results = sorted(
                    [
                        {"index": r.index, "score": getattr(r, "relevance_score", 0.0)}
                        for r in getattr(resp, "results", [])
                    ],
                    key=lambda x: x["score"],
                    reverse=True,
                )
                if top_n is not None:
                    results = results[: top_n]
                return results

        return CohereRerankerAdapter(client, cohere_model)

    together_model = model_name.split(":", 1)[1] if model_name.startswith("together:") else model_name
    from together import Together

    client = Together()

    class TogetherRerankerAdapter:
        def __init__(self, client_inst, model_str):
            self._client = client_inst
            self._model = model_str

        def rerank(self, *, query: str, documents: list, top_n: int | None = None):
            resp = self._client.rerank.create(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False,
            )
            results = sorted(
                [
                    {"index": r.index, "score": getattr(r, "relevance_score", 0.0)}
                    for r in getattr(resp, "results", [])
                ],
                key=lambda x: x["score"],
                reverse=True,
            )
            if top_n is not None:
                results = results[: top_n]
            return results

    return TogetherRerankerAdapter(client, together_model)