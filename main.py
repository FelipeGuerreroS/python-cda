import os
import re
import json
import time
import threading
import base64
import uuid
import logging
import ast
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from decimal import Decimal
from time import perf_counter
from typing import Any, Mapping, Dict, List, Set, Tuple, Optional
from urllib.parse import urlparse, urlunparse
from html import escape
import boto3
import concurrent.futures
import requests
from boto3.dynamodb.conditions import Key
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
    ConnectTimeoutError,
)

LOGGER = logging.getLogger(__name__)

_RE_COMILLAS = re.compile(r'(["“])(.+?)(["”])')   # "texto" o “texto”
_RE_NEGRITA  = re.compile(r'\*\*(.+?)\*\*')       # **texto**
REASONING_BLOCK_RE = re.compile(
    r'<\s*reasoning\b[^>]*>(.*?)</\s*reasoning\s*>',
    flags=re.IGNORECASE | re.DOTALL
)


def _configure_logging(level: str) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(message)s")
    LOGGER.setLevel(level)


def _log_print(*args: Any, **kwargs: Any) -> None:
    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)
    LOGGER.info(message)


# Reasignamos print localmente para mantener los mensajes y usar logging en Lambda.
print = _log_print  # type: ignore[assignment]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class Config:
    aws_region: str
    conversations_table: str
    questions_table: str
    kb_id: str
    haiku_model_id: str
    haiku_45_model_id: str
    sonnet_model_id: str
    sonnet_37_model_id: str
    sonnet_4_model_id: str
    sonnet_45_model_id: str
    titan_premier_model_id: str
    titan_express_model_id: str
    gpt_oss_120_model_id: str
    answer_max_tokens: int
    guardrail_id: str
    guardrail_ver: str
    trace: str
    anthropic_version: str
    categories: list[str]
    include_exec_times: bool
    kb_max_results: int
    kb_search_type: str
    s3_files_prefix: str
    invoke_timeout_seconds: float
    bedrock_runtime_fallback_region: str
    log_level: str


def load_config() -> Config:
    return Config(
        aws_region=_require_env("REGION"),
        conversations_table=_require_env("CONVERSATIONS_TABLE"),
        questions_table=_require_env("QUESTIONS_TABLE"),
        kb_id=_require_env("KB_ID"),
        haiku_model_id=_require_env("HAIKU_MODEL_ID"),
        haiku_45_model_id=_require_env("HAIKU_45_MODEL_ID"),
        sonnet_model_id=_require_env("SONNET_MODEL_ID"),
        sonnet_37_model_id=_require_env("SONNET_37_MODEL_ID"),
        sonnet_4_model_id=_require_env("SONNET_4_MODEL_ID"),
        sonnet_45_model_id=_require_env("SONNET_45_MODEL_ID"),
        titan_premier_model_id=_require_env("TITAN_PREMIER_MODEL_ID"),
        titan_express_model_id=_require_env("TITAN_EXPRESS_MODEL_ID"),
        gpt_oss_120_model_id=_require_env("GPT_OSS_120_MODEL_ID"),
        answer_max_tokens=_get_int_env("ANSWER_MAX_TOKENS", 6000),
        guardrail_id=os.environ.get("GUARDRAIL_ID", "sdsuv2s4z9op"),
        guardrail_ver=os.environ.get("GUARDRAIL_VER", "DRAFT"),
        trace=os.environ.get("TRACE", "ENABLED"),
        anthropic_version=os.environ.get("ANTHROPIC_VERSION", "bedrock-2023-05-31"),
        categories=os.environ.get("CATEGORIES", "").split(","),
        include_exec_times=_get_bool_env("INCLUDE_EXEC_TIMES", True),
        kb_max_results=_get_int_env("KB_MAX_RESULTS", 17),
        kb_search_type=os.environ.get("KB_SEARCH_TYPE", "HYBRID"),
        s3_files_prefix=os.environ.get("S3_FILES_PREFIX", "s3://files-cda-int/contenido_articulos/"),
        invoke_timeout_seconds=_get_float_env("INVOKE_TIMEOUT_SECONDS", 27.5),
        bedrock_runtime_fallback_region=os.environ.get(
            "BEDROCK_RUNTIME_FALLBACK_REGION",
            "us-west-2",
        ),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )

## FUNCIONES PARA MARCAR TIEMPOS DE EJECUCIÓN

def tstart() -> float:
    return perf_counter()

def mark(execution_times: dict, key: str, t0: float) -> None:
    execution_times[key] = round(perf_counter() - t0, 2)

## VARIABLES GLOBALES

CONFIG = load_config()
_configure_logging(CONFIG.log_level)

# ENV
AWS_REGION = CONFIG.aws_region
CONV_TABLE_NAME = CONFIG.conversations_table
QUESTIONS_TABLE_NAME = CONFIG.questions_table
KB_ID = CONFIG.kb_id

# MODELOS
HAIKU_MODEL_ID = CONFIG.haiku_model_id
HAIKU_45_MODEL_ID = CONFIG.haiku_45_model_id
SONNET_MODEL_ID = CONFIG.sonnet_model_id
SONNET_37_MODEL_ID = CONFIG.sonnet_37_model_id
SONNET_4_MODEL_ID = CONFIG.sonnet_4_model_id
SONNET_45_MODEL_ID = CONFIG.sonnet_45_model_id
TITAN_PREMIER_MODEL_ID = CONFIG.titan_premier_model_id
TITAN_EXPRESS_MODEL_ID = CONFIG.titan_express_model_id
GPT_OSS_120_MODEL_ID = CONFIG.gpt_oss_120_model_id

# OUTPUT IA
ANSWER_MAX_TOKENS = CONFIG.answer_max_tokens
GUARDRAIL_ID = CONFIG.guardrail_id
GUARDRAIL_VER = CONFIG.guardrail_ver
TRACE = CONFIG.trace
ANTHROPIC_VERSION = CONFIG.anthropic_version

# OTROS
CATEGORIES = CONFIG.categories  # DEPRECADO: SE DEJA POR SI CAMBIA A FUTURO
INCLUDE_EXEC_TIMES = CONFIG.include_exec_times

## CLIENTES AWS
dynamo              = boto3.resource('dynamodb', region_name=AWS_REGION)
bedrock_agent       = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_runtime     = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_usw2= boto3.client('bedrock-runtime', region_name=CONFIG.bedrock_runtime_fallback_region) 
s3_client           = boto3.client('s3', region_name=AWS_REGION)

#TABLAS DYNAMO
conversations_table = dynamo.Table(CONV_TABLE_NAME)
questions_table     = dynamo.Table(QUESTIONS_TABLE_NAME)

## MENSAJES PROTOTIPOS
MIN_VALID_CHARS = 1
S3_PREFIX = "s3://files-cda-int/contenido_dependencias"
error_msg = "Error"

_S3_RE = re.compile(
    r"""
    s3://[^\s'\"?#]+/        # prefijo y carpetas
    (?P<basename>[^/]+\.(?P<ext>csv|json))   # archivo + extensión
    (?:[?#][^\s'"]*)?        # opcional query/anchor
    """,
    re.IGNORECASE | re.VERBOSE
)

def parse_request(event: Mapping[str, Any]) -> Tuple[dict, Optional[str]]:
    """
    Devuelve (data_dict, err) donde data_dict al menos contiene 'question' si es posible.
    Pensado para AWS API Gateway HTTP API v2:
      - Tolera: JSON, x-www-form-urlencoded, texto plano.
      - Maneja cuerpos base64.
      - Si no hay body, intenta usar queryStringParameters.
    """

    if (
        isinstance(event, dict) 
        and 'body' not in event 
        and 'queryStringParameters' not in event 
        and 'question' in event
    ):
        return (event, None)

    # 1) Headers normalizados
    headers = {str(k).lower(): v for k, v in (event.get('headers') or {}).items()}

    # 2) Body (posible base64)
    body = event.get('body')

    if event.get('isBase64Encoded') and body is not None:
        try:
            # En v2 el body suele venir como str (base64); b64decode acepta str ASCII
            body = base64.b64decode(body).decode('utf-8', errors='replace')
        except Exception as e:
            return ({}, f"Base64 decode error: {e}")

    # 3) Si no hay body, intenta tirar de queryStringParameters (típico en GET)
    if body is None:
        qs = event.get('queryStringParameters') or {}
        if isinstance(qs, dict) and qs:
            return (qs, None)
        return ({}, None)

    # Asegúrate de tener str
    if not isinstance(body, str):
        body = str(body)

    ctype = headers.get('content-type', '')
    ctype_no_params = ctype.split(';', 1)[0].strip().lower()

    # 4) Si es JSON, parsea estrictamente
    if ctype_no_params == 'application/json':
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            return ({}, f"Invalid JSON: {e}")

        # Si no es dict (por ej. una lista), lo envolvemos
        if isinstance(payload, dict):
            return (payload, None)
        return ({"_raw": payload}, None)

    # 5) Si es form-urlencoded
    if ctype_no_params == 'application/x-www-form-urlencoded':
        parsed = urllib.parse.parse_qs(body, keep_blank_values=True)
        # Aplana: toma el primer valor
        flat = {k: (v[0] if isinstance(v, list) and v else '') for k, v in parsed.items()}
        return (flat, None)

    # 6) multipart/form-data: mejor esfuerzo (sin intentar parsearlo de verdad)
    if ctype_no_params == 'multipart/form-data':
        text = body.strip()
        if text:
            return ({"_raw": text}, None)
        return ({}, None)

    # 7) Caso genérico: trata el cuerpo como texto plano
    text = body.strip()
    if text:
        return ({"question": text}, None)

    return ({}, None)

def _select_bedrock_client(model_id: str):
    if model_id.startswith("openai.gpt-oss"):
        return bedrock_runtime
    return bedrock_runtime

def _invoke_chat(model_id, prompt_text, max_tokens, temperature, top_p=None, stop_sequences=None):
    # DIFERENCIACIÓN POR MODELO (Anthropic Claude / Amazon Titan / OpenAI OSS)
    is_anthropic = model_id.startswith("us.anthropic.")
    is_titan     = model_id.startswith("amazon.titan-text")
    is_openai    = model_id.startswith("openai.gpt-oss")

    accept       = "application/json"
    content_type = "application/json"

    system_prompt = """
    Eres un Asistente IA Profesional para LATAM (Agente LATAM), especializado en soporte operativo, normativo y documental de procesos internos de la aerolínea. Responde siempre con precisión, claridad y enfoque operativo.

    GLOSARIO:
    - Ancillaries = servicios adicionales / servicios complementarios.
    - EMD (Electronic Miscellaneous Document) = Documento electrónico usado para cobrar servicios adicionales (ancillaries). No es interlineal ni endosable. Puede ser asociado a un pasaje (EMD-A) o stand-alone (EMD-S). Validez: 1 año; personal e intransferible. Su uso depende estrictamente del servicio para el cual fue emitido. Los EMD-A se sincronizan con los cupones del pasaje; cambios en itinerario o pasaje rompen la sincronización y requieren reemisión. Ancillaries vendidos junto al pasaje se emiten desde Agente360; servicios especiales y ventas independientes se gestionan en herramientas backup como Allegro. Para ventas indirectas, consultar  'EMD – Atención Agencias de Viajes LATAM '.

    Tu objetivo: realizar lo que se te pida de forma impecablemente profesional, siguiendo políticas LATAM, manteniendo contexto operativo y evitando inferencias no respaldadas por normativa interna.
    """

    if is_anthropic:
        payload = {
            "anthropic_version": ANTHROPIC_VERSION,
            "system":            system_prompt,
            "max_tokens":        int(max_tokens),
            "temperature":       float(temperature) if temperature is not None else 0.7,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]}
            ]
        }

    elif is_titan:
        titan_input = f"System: {system_prompt}\nUser: {prompt_text}\nBot:"
        cfg = {"maxTokenCount": int(max_tokens)}
        if temperature is not None:
            cfg["temperature"] = float(temperature)
        if top_p is not None:
            cfg["topP"] = float(top_p)
        if stop_sequences:
            cfg["stopSequences"] = stop_sequences

        payload = {
            "inputText": titan_input,
            "textGenerationConfig": cfg
        }

    elif is_openai:
        # Formato OpenAI-like (Bedrock endpoint OpenAI-compatible)
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt_text}
            ],
            "max_completion_tokens": int(max_tokens),
            "temperature": float(temperature) if temperature is not None else 0.7

        }

    else:
        # Fallback a Titan-style
        payload = {
            "inputText": prompt_text,
            "textGenerationConfig": {
                "maxTokenCount": 256,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9
            }
        }

    # ——— Invocación con ruteo por región ———
    client = _select_bedrock_client(model_id)
    t_inv = time.time()
    try:
        resp = client.invoke_model(
            modelId     = model_id,
            body        = json.dumps(payload),
            contentType = content_type,
            accept      = accept
        )
    except getattr(bedrock_runtime, "exceptions", object()).ThrottlingException as e:
        print(f"[Bedrock Throttling] model={model_id} err={e}", flush=True)
        raise
    except (EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError) as e:
        print(f"[Bedrock Network/Timeout] model={model_id} err={e}", flush=True)
        raise
    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code")
        msg  = err.get("Message")
        print(f"[Bedrock ClientError] model={model_id} code={code} msg={msg}", flush=True)
        try:
            body = e.response.get("Body")
            if body:
                detail = body.read().decode("utf-8", errors="ignore")
                print(f"[Bedrock ClientError Body] {detail[:1000]}", flush=True)
        except Exception:
            pass
        raise
    except BotoCoreError as e:
        print(f"[Bedrock BotoCoreError] model={model_id} err={e}", flush=True)
        raise
    except Exception as e:
        print(f"[Bedrock UnknownError] model={model_id} err={e}", flush=True)
        raise

    took = time.time() - t_inv
    print(f"[Bedrock invoke_model OK] model={model_id} took={took:.2f}s", flush=True)

    data = json.loads(resp["body"].read())

    # ---- Parseo ANTHROPIC ----
    if is_anthropic:
        if "completion" in data:
            return data["completion"]
        if "messages" in data and data["messages"]:
            for block in data["messages"][0].get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
        if "content" in data and isinstance(data["content"], list):
            for block in data["content"]:
                if block.get("type") == "text":
                    return block.get("text", "")
        return ""

    # ---- Parseo TITAN ----
    if is_titan:
        if "results" in data and data["results"]:
            return data["results"][0].get("outputText", "")
        return ""

    # ---- Parseo OPENAI OSS (OpenAI-like) ----
    if is_openai:
        # Puede venir como chat.completions o completions legacy (text)
        if "choices" in data and data["choices"]:
            ch = data["choices"][0]
            if isinstance(ch, dict):
                # chat
                msg = ch.get("message")
                if msg and isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                # text
                if "text" in ch:
                    return ch["text"]
        return ""

    # ---- Fallback ----
    if "results" in data and data["results"]:
        return data["results"][0].get("outputText", "")
    return ""

def _invoke_chat_with_timeout(
    model_id,
    prompt_text,
    max_tokens,
    temperature,
    top_p=None,
    stop_sequences=None,
    timeout_seconds=None,
):
    """
    Envuelve _invoke_chat con un timeout duro.
    - Si el modelo responde antes del timeout -  devuelve (respuesta, False)
    - Si se pasa del timeout -  devuelve ("Intermitencia de Servicio", True)
    """
    result = {"value": None}
    error = {"exc": None}

    def runner():
        try:
            result["value"] = _invoke_chat(
                model_id=model_id,
                prompt_text=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences
            )
        except Exception as e:
            error["exc"] = e

    if timeout_seconds is None:
        timeout_seconds = CONFIG.invoke_timeout_seconds

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout_seconds)

    # 1) Timeout
    if t.is_alive():
        # No matamos el hilo: cuando la Lambda termine, se congela todo el proceso.
        return "Intermitencia de Servicio", True

    # 2) Error real del modelo
    if error["exc"] is not None:
        # Aquí decides si quieres propagar el error o transformarlo:
        return "Intermitencia de Servicio", True

    # 3) Respuesta OK
    return result["value"] or "", False

def _normalize_language_filter(language: Optional[str]) -> str:
    if not language:
        return "español"
    normalized = language.strip().lower()
    if normalized in ("es", "español"):
        return "español"
    if normalized in ("pt", "portugues"):
        return "portugues"
    return normalized


def retrieve_kb(txt: str, language: Optional[str]) -> list[dict]:
    normalized_language = _normalize_language_filter(language)
    metadata_filter = {
        "equals": {
            "key": "idioma",
            "value": normalized_language,
        }
    }
    kb_resp = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": CONFIG.kb_max_results,
                "overrideSearchType": CONFIG.kb_search_type,
                "filter": metadata_filter,
            }
        },
        retrievalQuery={"text": txt}
    )
    return kb_resp.get("retrievalResults", [])

def normalizar_texto_para_match(texto: str) -> str:
    if not texto:
        return ""

    return re.sub(r"\s+", " ", texto.strip().lower())

def normalize_results(results, source_label, max_description_len=15000):
    normalized = []
    for r in results:
        metadata = r.get("metadata", {})
        location = r.get("location", {})
        s3_loc = location.get("s3Location", {})
        doc_id = s3_loc.get("uri")  # Ajusta si usas otro campo como id

        description = metadata.get("description")
        if isinstance(description, str) and len(description) > max_description_len:
            description = description[:max_description_len] + "..."

        normalized.append({
            "doc_id": doc_id,
            "score": r.get("score"),
            "title": metadata.get("title"),
            "description": metadata.get("description"),
            "idioma": metadata.get("idioma"),
            "source": source_label,
            "raw": r
        })
    return normalized

def same_top_doc(question_chunks, rephrased_chunks, min_score=0.49, debug=False):
    """
    Devuelve el doc consensuado solo si:
      - ambos retrieves tienen top-1 con el mismo doc_id
      - y al menos uno de los scores del top-1 >= min_score
    Si no, devuelve None.
    """
    if not question_chunks or not rephrased_chunks:
        return None

    top_q = question_chunks[0]
    top_r = rephrased_chunks[0]

    doc_id_q = top_q.get("doc_id")
    doc_id_r = top_r.get("doc_id")

    # Si no es el mismo artículo, no hay consenso
    if not doc_id_q or doc_id_q != doc_id_r:
        if debug:
            print("[CONSENSO] Sin consenso de doc_id en top-1")
        return None

    # Scores (si vienen None, los tomamos como 0.0)
    score_q = top_q.get("score") or 0.0
    score_r = top_r.get("score") or 0.0
    max_score = max(score_q, score_r)

    if debug:
        print(
            f"[CONSENSO] doc_id={doc_id_q} "
            f"score_q={score_q:.3f} score_r={score_r:.3f} "
            f"max_score={max_score:.3f} min_score={min_score}"
        )

    # Si ninguno de los dos supera el umbral, NO usamos consenso
    if max_score < min_score:
        if debug:
            print("[CONSENSO] Mismo artículo, pero scores por debajo del umbral -> se descarta consenso")
        return None

    if debug:
        print("[CONSENSO] Mismo artículo y score suficiente -> se acepta consenso")

    return top_q

def merge_results(question_chunks, rephrased_chunks):
    merged = defaultdict(lambda: {
        "doc_id": None,
        "title": None,
        "description": None,
        "topics": None,
        "not_covered": None,
        "intents": None,
        "idioma": None,
        "from_original": None,
        "from_rephrased": None,
        "doc": None,  # para guardar el doc normalizado
    })

    for idx, item in enumerate(question_chunks):
        d = merged[item["doc_id"]]
        d["doc_id"] = item["doc_id"]
        if item.get("title") is not None:
            d["title"] = item["title"]
        if item.get("context_short") is not None:
            d["description"] = item["context_short"]
        if item.get("context_bullets") is not None:
            d["topics"] = item["context_bullets"]
        if item.get("not_covered") is not None:
            d["not_covered"] = item["not_covered"]
        if item.get("intents") is not None:
            d["intents"] = item["intents"]
        if item.get("idioma") is not None:
            d["idioma"] = item["idioma"]
        d["from_original"] = {
            "rank": idx + 1,
            "score": item["score"]
        }
        if d["doc"] is None:
            d["doc"] = item

    for idx, item in enumerate(rephrased_chunks):
        d = merged[item["doc_id"]]
        d["doc_id"] = item["doc_id"]
        if item.get("title") is not None and d["title"] is None:
            d["title"] = item["title"]
        if item.get("context_short") is not None and d["context_short"] is None:
            d["description"] = item["context_short"]
        if item.get("context_bullets") is not None:
            d["topics"] = item["context_bullets"]
        if item.get("not_covered") is not None:
            d["not_covered"] = item["not_covered"]
        if item.get("intents") is not None:
            d["intents"] = item["intents"]
        if item.get("idioma") is not None and d["idioma"] is None:
            d["idioma"] = item["idioma"]
        d["from_rephrased"] = {
            "rank": idx + 1,
            "score": item["score"]
        }
        if d["doc"] is None:
            d["doc"] = item

    return list(merged.values())

def formatear_texto(texto: str) -> str:
    # 1) Comillas dobles (ASCII o tipográficas) -> negrita + cursiva
    #    Usamos grupo 2: el contenido dentro de las comillas
    texto = _RE_COMILLAS.sub(r'<b><i>\2</i></b>', texto)

    # 2) Doble asterisco -> negrita
    texto = _RE_NEGRITA.sub(r'<b>\1</b>', texto)

    return texto

def parsear_respuesta_ia(respuesta_ia: str):
    """
    Convierte una respuesta tipo '(2)[1,2]' en:
    chosen_doc_id = '2'
    chosen_flujos = [1, 2]
    """
    match = re.match(r'^\s*\(([^)]+)\)\s*\[([^\]]*)\]\s*$', respuesta_ia)

    if not match:
        raise ValueError(f"Formato inválido devuelto por IA: {respuesta_ia}")

    chosen_doc_id = match.group(1).strip()
    flujos_raw = match.group(2).strip()

    if flujos_raw:
        chosen_flujos = [
            int(x.strip())
            for x in flujos_raw.split(",")
            if x.strip()
        ]
    else:
        chosen_flujos = []

    return chosen_doc_id, chosen_flujos

def pick_by_heuristics(
    merged_docs,
    min_avg_score=0.75,
    max_rank_allowed=2,
    debug=False
):
    candidates = []
    for d in merged_docs:
        fo = d.get("from_original")
        fr = d.get("from_rephrased")
        if fo and fr:
            avg_score = (fo["score"] + fr["score"]) / 2.0
            max_rank = max(fo["rank"], fr["rank"])

            if debug:
                print(
                    f"[HEURISTIC] doc_id={d.get('doc_id')} "
                    f"avg_score={avg_score:.3f} "
                    f"ranks=(orig={fo['rank']}, rephr={fr['rank']})"
                )

            if avg_score >= min_avg_score and max_rank <= max_rank_allowed:
                candidates.append((avg_score, -max_rank, d))

    if not candidates:
        if debug:
            print("[HEURISTIC] Ningún documento cumple los criterios de aceptación.")
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    best_merged = candidates[0][2]

    if debug:
        print(
            f"[HEURISTIC] Documento seleccionado por heurística: "
            f"doc_id={best_merged.get('doc_id')}"
        )

    # devolvemos el doc normalizado que tiene el "raw"
    return best_merged.get("doc") or best_merged

def build_llm_candidates_payload(merged_docs, max_candidates=15):
    # Ordenar por mejor score (tomando el mejor de original/rephrased)
    def best_score(d):
        scores = []
        if d.get("from_original"):
            scores.append(d["from_original"]["score"])
        if d.get("from_rephrased"):
            scores.append(d["from_rephrased"]["score"])
        return max(scores) if scores else 0.0

    sorted_docs = sorted(merged_docs, key=lambda d: best_score(d), reverse=True)
    selected = sorted_docs[:max_candidates]

    llm_docs = []
    for i, d in enumerate(selected, start=1):
        fo = d.get("from_original")
        fr = d.get("from_rephrased")

        llm_docs.append({
            "id": str(i),
            "doc_id": d["doc_id"],
            "doc": d.get("doc"),
            "title": d.get("title"),
            "description": d.get("description"),
            "idioma": d.get("idioma"),
            "original_rank": fo["rank"] if fo else None,
            "original_score": fo["score"] if fo else None,
            "rephrased_rank": fr["rank"] if fr else None,
            "rephrased_score": fr["score"] if fr else None
        })
    return llm_docs

def find_doc_by_id(all_normalized_results, chosen_id):
    for item in all_normalized_results:
        print("doc_id")
        print(item["doc_id"])
        if item["doc_id"] == chosen_id:
            return item
    return None

def normalizar_url_para_match(url: str) -> str:
    """
    Normaliza URLs para comparar:
    - baja netloc a minúsculas
    - elimina fragmentos tipo #
    - elimina slash final
    """
    if not url:
        return ""

    url = url.strip().strip('"').strip("'").strip()

    # Quitar caracteres típicos que pueden quedar pegados por regex
    url = url.rstrip(").,;] ")

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path

    return urlunparse((
        scheme,
        netloc,
        path,
        "",
        parsed.query,
        ""  # eliminamos fragment/hash
    ))


def extraer_flujos_desde_metadata(meta: dict) -> List[dict]:
    """
    Convierte metadata['urls'] en una lista de flujos numerados.

    Ejemplo de entrada:
    [
        'https://...;Flujo Emisión Exceso de Equipaje',
        'https://...;Flujo Emisión Oversize de Equipaje'
    ]

    Salida:
    [
        {'numero': 1, 'url': 'https://...', 'url_norm': 'https://...', 'titulo': '...'},
        {'numero': 2, 'url': 'https://...', 'url_norm': 'https://...', 'titulo': '...'}
    ]
    """
    urls = meta.get("urls", []) or []

    flujos = []

    for idx, item in enumerate(urls, start=1):
        if not isinstance(item, str):
            continue

        item = item.strip().strip('"').strip("'")

        if ";" not in item:
            continue

        url, titulo = item.split(";", 1)

        url = url.strip()
        titulo = titulo.strip()

        flujos.append({
            "numero": idx,
            "url": url,
            "url_norm": normalizar_url_para_match(url),
            "titulo": titulo
        })

    return flujos


def obtener_flujos_seleccionados(meta: dict, chosen_flujos: List[int]) -> List[dict]:
    """
    Devuelve solo los flujos elegidos por la IA.
    """
    flujos = extraer_flujos_desde_metadata(meta)
    chosen_set = set(chosen_flujos)

    return [
        flujo for flujo in flujos
        if flujo["numero"] in chosen_set
    ]

def filtrar_flujos_no_seleccionados_en_texto(
    sel_text: str,
    selected_flow_urls_norm: Set[str],
    selected_flow_titles_norm: Set[str] = None,
    window_before: int = 1200,
    window_after: int = 300
) -> str:
    """
    Elimina los marcadores [FLUJO] s3://... que NO correspondan
    a los flujos seleccionados por la IA.

    Hace match por:
    1) URL cercana al marcador [FLUJO]
    2) Título cercano al marcador [FLUJO]
    """

    if selected_flow_titles_norm is None:
        selected_flow_titles_norm = set()

    flujo_pattern = re.compile(
        r'\(?\s*\[FLUJO\]\s*(?P<uri>s3://.*?\.(?:csv|txt|json))\s*\)?',
        flags=re.IGNORECASE | re.UNICODE
    )

    url_pattern = re.compile(
        r'https?://[^\s\)\]\}]+',
        flags=re.IGNORECASE | re.UNICODE
    )

    partes = []
    last_pos = 0

    for match in flujo_pattern.finditer(sel_text):
        start, end = match.span()

        texto_antes = sel_text[max(0, start - window_before):start]
        texto_despues = sel_text[end:min(len(sel_text), end + window_after)]
        texto_cercano = texto_antes + " " + texto_despues
        texto_cercano_norm = normalizar_texto_para_match(texto_cercano)

        flujo_es_seleccionado = False

        # 1) Match por URL cercana
        urls_cercanas = url_pattern.findall(texto_cercano)

        for url_cercana in urls_cercanas:
            url_cercana_norm = normalizar_url_para_match(url_cercana)

            if url_cercana_norm in selected_flow_urls_norm:
                flujo_es_seleccionado = True
                break

        # 2) Match por título cercano
        if not flujo_es_seleccionado:
            for titulo_norm in selected_flow_titles_norm:
                if titulo_norm and titulo_norm in texto_cercano_norm:
                    flujo_es_seleccionado = True
                    break

        partes.append(sel_text[last_pos:start])

        if flujo_es_seleccionado:
            partes.append(sel_text[start:end])
        else:
            partes.append("")

        last_pos = end

    partes.append(sel_text[last_pos:])

    return "".join(partes)

def extraer_uris_flujo_presentes(sel_text: str) -> List[str]:
    """
    Extrae las URIs s3 de flujos que siguen presentes en el texto.
    """
    pattern = re.compile(
        r'\[FLUJO\]\s*(s3://.*?\.(?:csv|txt|json))',
        flags=re.IGNORECASE | re.UNICODE
    )

    return pattern.findall(sel_text)

def normalizar_links(lista_links):
    """
    Recibe una lista con formato 'url;titulo' y devuelve una lista sin duplicados,
    normalizando solo la URL y preservando el título.
    """
    links_limpios = {}

    for item in lista_links:
        if not isinstance(item, str):
            continue

        item = item.strip().strip('"').strip("'")

        if ';' not in item:
            continue

        url, titulo = item.split(';', 1)
        url = url.strip()
        titulo = titulo.strip()

        parsed = urlparse(url)

        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip('/') if parsed.path != '/' else parsed.path

        normalized_url = urlunparse((
            parsed.scheme,
            netloc,
            path,
            '',
            parsed.query,
            parsed.fragment
        ))

        # Deduplicar por URL normalizada, preservando el primer título encontrado
        if normalized_url not in links_limpios:
            links_limpios[normalized_url] = titulo

    return [
        f"{url};{titulo}"
        for url, titulo in links_limpios.items()
    ]

def parsear_link_y_titulo(valor: Any) -> Dict[str, str]:
    if not isinstance(valor, str):
        return {"url": "", "title": ""}

    texto = valor.strip().strip('"').strip("'")
    if not texto:
        return {"url": "", "title": ""}

    if ";" in texto:
        url, titulo = texto.split(";", 1)
        return {
            "url": url.strip(),
            "title": titulo.strip() or "Sin Titulo",
        }

    return {
        "url": texto,
        "title": "Sin Titulo",
    }


def normalizar_url(url: str) -> str:
    if not isinstance(url, str):
        return ""

    url = url.strip().strip('"').strip("'")
    if not url:
        return ""

    parsed = urlparse(url)

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path not in ("", "/") else parsed.path

    return urlunparse((scheme, netloc, path, "", parsed.query, ""))

def extraer_titulos_links(lista_links):
    """
    Recibe una lista de links con formato 'url;titulo'
    y devuelve solo los títulos.
    """
    titulos = []

    for item in lista_links:
        if not isinstance(item, str):
            continue

        item = item.strip().strip('"').strip("'")

        if ';' not in item:
            continue

        _, titulo = item.split(';', 1)
        titulo = titulo.strip()

        if titulo:
            titulos.append(titulo)

    return titulos

def convertir_links_a_html(links):
    if isinstance(links, str):
        links = ast.literal_eval(links)

    resultado = []

    for item in links:
        if ';' not in item:
            continue

        url, titulo = item.split(';', 1)
        url = url.strip()
        titulo = titulo.strip()

        html_link = (
            f'<a href="{escape(url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">'
            f'{escape(titulo)}'
            f'</a>'
        )

        resultado.append(html_link)

    return resultado

def strip_s3_scheme(value: str) -> str:
    if not value:
        return ""
    value = str(value).strip()
    value = re.sub(r"^s3:/+", "", value)
    return value.lstrip("/")

def normalize_s3_uri(uri: str) -> str:
    if not uri:
        return ""

    uri = str(uri).strip()

    # elimina prefijo s3:/, s3://, s3:////, etc. solo al inicio
    uri = re.sub(r"^s3:/+", "", uri)

    # elimina slashes repetidos en la parte restante
    uri = re.sub(r"/{2,}", "/", uri.lstrip("/"))

    return f"s3://{uri}"

def fetch_s3_text(uri: str) -> str:
    parsed = urlparse(uri)
    # Caso s3://
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return obj['Body'].read().decode('utf-8')
    
    # Caso HTTPS S3 pública o privada
    if parsed.scheme in ("http", "https") and ".s3." in parsed.netloc:
        # extrae bucket de 'bucket.s3.[region].amazonaws.com'
        bucket = parsed.netloc.split(".s3.")[0]
        key = parsed.path.lstrip("/")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return obj['Body'].read().decode('utf-8')
    
    # Fallback a cualquier otro HTTP
    resp = requests.get(uri, timeout=5)
    resp.raise_for_status()
    return resp.text

def normalize_text(raw_txt: str) -> str:
    return re.sub(r"\s+", " ", raw_txt.replace("\n", " ")).strip()

def extraer_uris_por_tipo(texto: str, tipo: str) -> List[str]:
    """
    Extrae URIs s3 asociadas a un tipo específico: TABLA o FLUJO.
    Busca patrones como:
    [TABLA] s3://...
    [FLUJO] s3://...
    """
    pattern = re.compile(
        rf'\[{re.escape(tipo)}\]([\s\S]*?)(s3://.*?\.(?:csv|txt|json))',
        flags=re.IGNORECASE | re.UNICODE
    )

    uris = []

    for match in pattern.finditer(texto):
        uri = match.group(2).strip()
        uris.append(uri)

    # deduplicar preservando orden
    return list(dict.fromkeys(uris))


def cargar_dependencias(uris: List[str]):
    """
    Descarga el contenido de las dependencias s3.
    Retorna metadata_list y uri_to_content.
    """
    metadata_list = []
    uri_to_content: Dict[str, str] = {}

    for uri in uris:
        try:
            raw_txt, used_uri = fetch_with_fallback(uri)
            txt = normalize_text(raw_txt)
            full_uri = normalize_s3_uri(used_uri)

            metadata_list.append({
                "uri": full_uri,
                "content": txt
            })

            uri_to_content[full_uri] = txt

        except Exception:
            candidates = _build_s3_candidates(uri)
            fallback_uri = normalize_s3_uri(candidates[-1] if candidates else uri)

            metadata_list.append({
                "uri": fallback_uri,
                "content": "No encontrado: " + fallback_uri
            })

            uri_to_content[fallback_uri] = "No encontrado: " + fallback_uri

            continue

    return metadata_list, uri_to_content


def inyectar_dependencias_por_tipo(
    texto: str,
    uri_to_content: Dict[str, str],
    tipo: str
) -> str:
    """
    Inyecta dependencias de un solo tipo: TABLA o FLUJO.
    No toca otros tipos.
    """

    pattern = re.compile(
        rf'\[({re.escape(tipo)})\]([\s\S]*?)(s3://.*?\.(?:csv|txt|json))',
        flags=re.IGNORECASE | re.UNICODE
    )

    def replace_match(m: re.Match) -> str:
        kind = m.group(1).upper()
        middle = m.group(2) or ""
        uri_raw = m.group(3).strip()

        uri_norm = normalize_s3_uri(uri_raw)
        content = uri_to_content.get(uri_norm, "")

        if not content:
            return f'[{kind}]{middle}{uri_raw} [NO ENCONTRADO EN METADATA]'

        fin_tag = end_tag_for_uri(uri_norm, kind)

        return f'[{kind}]{middle}{{ {content} }} [{fin_tag}]'

    return pattern.sub(replace_match, texto)


def is_valid_text(raw_txt: str, min_chars: int = MIN_VALID_CHARS) -> bool:
    if not raw_txt:
        return False
    txt = normalize_text(raw_txt)
    return len(txt) >= min_chars


def _build_s3_candidates(uri: str, prefix: str = S3_PREFIX) -> list[str]:
    normalized_input = normalize_s3_uri(uri)
    if not normalized_input:
        return []

    normalized_prefix = normalize_s3_uri(prefix).rstrip("/")

    input_no_scheme = strip_s3_scheme(normalized_input)
    prefix_no_scheme = strip_s3_scheme(normalized_prefix)

    if input_no_scheme.startswith(prefix_no_scheme + "/"):
        short_candidate = input_no_scheme[len(prefix_no_scheme):].lstrip("/")
    else:
        short_candidate = input_no_scheme

    first_candidate = normalize_s3_uri(f"{normalized_prefix}/{short_candidate}")
    candidates = [first_candidate]

    if normalized_input != first_candidate:
        candidates.append(normalized_input)

    return candidates


def fetch_with_fallback(uri: str, prefix: str = S3_PREFIX, min_chars: int = MIN_VALID_CHARS):
    candidates = _build_s3_candidates(uri, prefix=prefix)
    errors = []

    for candidate in candidates:
        try:
            raw_txt = fetch_s3_text(candidate)
            if is_valid_text(raw_txt, min_chars=min_chars):
                return raw_txt, candidate

            errors.append(f"Contenido inválido o demasiado corto en: {candidate}")

        except Exception as e:
            errors.append(f"Error en {candidate}: {e}")

    raise ValueError("No se encontró contenido válido. " + " | ".join(errors))

def end_tag_for_uri(uri: str, fallback_kind: str) -> str:
    """Devuelve el tag de cierre en función de la extensión."""
    ext = uri.lower().rsplit('.', 1)[-1] if '.' in uri else ''
    if ext == 'csv':
        return 'FIN TABLA'
    if ext in ('json', 'txt'):
        return 'FIN FLUJO'
    return f'FIN {fallback_kind}'

def remove_reasoning_blocks(text: str) -> str:
    text = re.sub(
        r'<\s*reasoning\b[^>]*>.*?</\s*reasoning\s*>',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    # 2) Por si llega suelto un tag de apertura/cierre sin pareja
    text = re.sub(r'</?\s*reasoning\b[^>]*>', '', text, flags=re.IGNORECASE)

    # 3) Limpieza de comillas envolventes accidentales y espacios
    return text.strip().strip('"').strip("'").strip()

def split_reasoning_blocks(text: str) -> Tuple[str, List[str]]:
    # 1) Extraer el contenido interno de todos los bloques <reasoning>...</reasoning>
    reasoning_blocks = [
        match.group(1).strip()
        for match in REASONING_BLOCK_RE.finditer(text)
    ]

    # 2) Remover esos bloques del texto original
    clean_text = REASONING_BLOCK_RE.sub('', text)

    # 3) Por si llegan tags sueltos sin pareja
    clean_text = re.sub(
        r'</?\s*reasoning\b[^>]*>',
        '',
        clean_text,
        flags=re.IGNORECASE
    )

    # 4) Limpieza final
    clean_text = clean_text.strip().strip('"').strip("'").strip()

    return clean_text, reasoning_blocks

def remove_according_phrases(text: str) -> str:
    frases = [
        r"seg[uú]n\s+el\s+flujo",
        r"seg[uú]n\s+la\s+documentaci[oó]n",
        r"seg[uú]n\s+la\s+informaci[oó]n(?:\s+proporcionada)?",
        r"seg[uú]n\s+el\s+contexto",
        r"de acuerdo a la documentaci[oó]n",
        r"de acuerdo a la informaci[oó]n(?:\s+proporcionada)?",
        r"como se indica en la documentaci[oó]n",
        r"de acuerdo (?:al|con el) contexto",
        r"informaci[oó]n adicional relevante",
        r"informaci[oó]n relevante adicional",
        r"informaci[oó]n adicional",
        r"informaci[oó]n relevante",
        r"se\s+infier[eé]\s+que",
        r"se\s+infier[eé]\s+que,",
        r"la\s+informaci[oó]n\s+indica\s+que",
        r"seg[uú]n\s+la\s+informaci[oó]n\s+disponible",
        r"seg[uú]n\s+la\s+informaci[oó]n\s+disponible,",
        r"seg[uú]n\s+la\s+documentaci[oó]n\s+disponible",
        r"seg[uú]n\s+la\s+documentaci[oó]n\s+disponible,",
        r",\s*seg[uú]n\s+se\s+establece\s+expl[ií]citamente\s+en\s+la\s+informaci[oó]n\s+proporcionada",
        r",\s*seg[uú]n\s+se\s+establece\s+expl[ií]citamente\s+en\s+la\s+informaci[oó]n\s+proporcionada,",
    ]

    for frase in frases:
        pattern = rf"{frase}\s*(?:,|:)?\s*"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([,.:;])", r"\1", text)

    return text

	
def fetch_first_valid_text(uri: str, min_chars: int = MIN_VALID_CHARS):
    candidates = [
        f"{S3_PREFIX}{uri}",
        uri,
    ]
    last_error = None
    for candidate in candidates:
        try:
            raw_txt = fetch_s3_text(candidate)
            if is_valid_text(raw_txt, min_chars=min_chars):
                return candidate, raw_txt
        except Exception as e:
            last_error = e
    return None, None

def remove_resumen_tag(text: str) -> str:
    return re.sub(
        r'(\[\s*/?\s*(?:RESUMEN|RESUMÉN|RESUMN|RESUME)\s*\]|<\s*/?\s*(?:RESUMEN|RESUMÉN|RESUMN|RESUME)\s*>)',
        '',
        text,
        flags=re.IGNORECASE
    )

def clean_newlines(text):
    if text:
        return text.replace("\\n", "\n").replace("\n\n", "\n\n").strip()
    return ""

def clean_response_text(text: str) -> str:
    return remove_according_phrases(clean_newlines(text))

def format_to_html(raw_text: str) -> str:
    """
    Limpia y convierte un texto crudo con numeración, guiones y **bold**
    a HTML semántico con listas y etiquetas, manejando casos con y sin
    título antes de subviñetas.
    """
    if not raw_text:
        return ""

    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", raw_text.strip())

    html_parts = []

    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        has_numbered = re.search(r"(?m)^\d+\.\s", block) is not None
        if has_numbered:
            # Extraer cada paso numerado (línea que inicia con "N. ")
            items = re.findall(
                r"(?ms)^\d+\.\s*(.+?)(?=(?:^\d+\.\s)|\Z)",
                block
            )
            if not items:
                continue

            html_parts.append("<ol>")
            for it in items:
                it = it.strip()

                # Separar posibles subviñetas que comienzan con "-"
                # (solo consideramos los '-' al inicio de línea)
                subs = [s.strip() for s in re.split(r"(?m)^\s*-\s+", it) if s.strip()]

                # Caso A: el ítem NO tiene subviñetas
                if len(subs) == 0:
                    # Nada que mostrar (evitar li vacío): usamos el texto completo
                    if it:
                        html_parts.append(f"<li>{it}</li>")
                    continue
                if len(subs) == 1:
                    # Un solo bloque de texto (sin subviñetas reales)
                    html_parts.append(f"<li>{subs[0]}</li>")
                    continue

                # Caso B: hay varias partes. Determinar si el ítem empieza con '-'
                starts_with_dash = bool(re.match(r"^\s*-\s+", it))
                if starts_with_dash:
                    # No hay “título” antes de las subviñetas: renderizar solo la sublista
                    html_parts.append("<li><ul>")
                    for s in subs:
                        html_parts.append(f"<li>{s}</li>")
                    html_parts.append("</ul></li>")
                else:
                    # Hay un “título” (subs[0]) + subviñetas (subs[1:])
                    html_parts.append(f"<li>{subs[0]}<ul>")
                    for s in subs[1:]:
                        html_parts.append(f"<li>{s}</li>")
                    html_parts.append("</ul></li>")
            html_parts.append("</ol>")
        else:
            # --- 2) Bloque sin numeración: tratar como lista de viñetas si hay líneas con '-' ---
            bullet_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            # ¿Hay al menos una línea que empiece con "- "?
            if any(re.match(r"^-{1}\s+", ln) for ln in bullet_lines):
                html_parts.append("<ul>")
                for ln in bullet_lines:
                    m = re.match(r"^-{1}\s+(.*)", ln)
                    if m:
                        html_parts.append(f"<li>{m.group(1).strip()}</li>")
                    else:
                        # Línea que no empieza con '-', la mostramos como párrafo dentro de la lista
                        html_parts.append(f"<li>{ln}</li>")
                html_parts.append("</ul>")
            else:
                # Texto suelto
                html_parts.append(f"<p>{block}</p>")

    return "\n".join(html_parts)

def format_resumen(resumen: str) -> str:
    if not resumen:
        return ""

    text = resumen.strip()

    # Paso 1: Convertir negritas Markdown a HTML
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)

    

    # Paso 2: Normalizar listas numeradas en línea

    text = re.sub(
        r'(?<!\n)(?:;\s*|\s+)(\d+[.)]\s)',
        r'\n\1',
        text
    )

    # Normalizar listas numeradas en línea
    text = re.sub(
        r'(?<!\n)(?:;\s*|\s+)(\(\d+\)\s)',
         r'\n\1',
        text
    )

    # Normalizar bullet points en línea: ". - Texto" → salto + "- Texto 
    text = re.sub(
        r'(?<=[.:])\s+-\s+',
        r'\n- ',
        text
    )

    # Paso 3: Separar en bloques por doble salto de línea
    blocks = re.split(r"\n\s*\n", text)

    html_parts = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Paso 4: Detectar lista numerada
        has_numbered = re.search(r"(?m)^\s*(?:\d+[.)]\s|\(\d+\)\s)", block) is not None

        if has_numbered:
            intro_match = re.match(r"^(.*?)(?=\s*(?:\d+[.)]\s|\(\d+\)\s))", block, re.DOTALL)
            if intro_match:
                intro = intro_match.group(1).strip()
                if intro:
                    html_parts.append(f"{intro}<br>")

            items = re.findall(
                r"(?ms)(?:\d+[.)]|\(\d+\))\s*(.+?)(?=(?:(?:\d+[.)]|\(\d+\))\s)|\Z)",
                block
            )

            if items:
                html_parts.append("<ol>")
                for item in items:
                    item = item.strip()
                    item = re.sub(r";\s*$", "", item)
                    if item:
                        html_parts.append(f"<li>{item}</li>")
                html_parts.append("</ol>")

        # Paso 5: Detectar bullet points
        elif re.search(r"(?m)^\s*[-•]\s+", block):
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]

            intro_lines = []
            bullet_start = 0
            for i, ln in enumerate(lines):
                if re.match(r"^[-•]\s+", ln):
                    bullet_start = i
                    break
                intro_lines.append(ln)

            if intro_lines:
                html_parts.append(f"{' '.join(intro_lines)}<br>")

            html_parts.append("<ul>")
            for ln in lines[bullet_start:]:
                m = re.match(r"^[-•]\s+(.*)", ln)
                if m:
                    html_parts.append(f"<li>{m.group(1).strip()}</li>")
                else:
                    html_parts.append(f"<li>{ln}</li>")
            html_parts.append("</ul>")

        else:
            # Paso 6: Párrafo normal
            lines = block.splitlines()
            paragraph = "<br>".join(ln.strip() for ln in lines if ln.strip())
            html_parts.append(f"<p>{paragraph}</p>")

    return "\n".join(html_parts)

def convertir_links_html(resumen: str) -> str:
    patron_url = re.compile(
        r'(?<!href=")'                  # no inmediatamente después de href="
        r'((?:https?://|www\.)[^\s<]+)',# URL hasta espacio o '<'
        re.IGNORECASE
    )

    # Conjunto de puntuación que queremos dejar fuera del <a>
    PUNTO_FINAL = r'.,;:!?)\]]'

    def reemplazar(match):
        url_completa = match.group(1)

        # Separa puntuación final (si existe) para dejarla fuera del link
        trail = ""
        while url_completa and url_completa[-1] in PUNTO_FINAL:
            trail = url_completa[-1] + trail
            url_completa = url_completa[:-1]

        # Normaliza href (si empieza con www, anteponer http://)
        href = url_completa if url_completa.startswith(('http://', 'https://')) else f'http://{url_completa}'

        enlace = (
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">este enlace</a>'
        )
        return enlace + trail

    return patron_url.sub(reemplazar, resumen)

def lambda_handler(event: Mapping[str, Any], context: Any) -> dict:
    """
    AWS Lambda handler para procesar preguntas, recuperar contexto y generar respuestas.
    """
    t0 = tstart()
    execution_times = {}
    print(f"----------INICIO - EVENTO RECIBIDO----------")
    print(f"{event}")
    print(f"----------FIN - EVENTO RECIBIDO----------")
    print("***********************************************************")
    data, perr = parse_request(event)

    question = (data or {}).get('question')
    print(f"PREGUNTA: ",question)
    session_id = (data or {}).get('session_id')
    print(f"ID DE INTERACCIÓN: ",session_id)
    language = (data or {}).get('language') or 'español'
    print(f"IDIOMA: ",language)
    name = (data or {}).get('name')
    print(f"NOMBRE AGENTE: ",name)
    email = (data or {}).get('email')
    print(f"EMAIL: ",email)

    if not question:
        print("No 'question' in body")
        msg = "No 'question' in body"
        if perr:
            msg += f" ({perr})"
        return {"statusCode": 400, "body": json.dumps({"error": msg})}
    
    mark(execution_times, 'eventParser', t0)
    t1 = tstart()
    print(f"----------INICIO - RECUPERACION HISTORIAL----------")
    item = None
    if session_id:
        try:
            resp = conversations_table.get_item(Key={'user_id': session_id})
            item = resp.get('Item')
        except ClientError:
            item = None
    else:
        session_id = str(uuid.uuid4())
    history = item.get('history', []) if item else []
    mark(execution_times, 'getUserHistory', t1)
    print(f"----------FIN - RECUPERACION HISTORIAL----------")
    print("***********************************************************")
    t2 = tstart()
    print(f"----------INICIO - REFRASEO PREGUNTA----------")
    rephrase_context = [question] ##history[-2:] + [question]
    rephrase_prompt = (
        "Contexto:\n"
        + "\n".join(rephrase_context)
        + "\n\nCorrige la ortografía y puntuación de la consulta anterior (tildes, comas y signos de interrogación) para formar una sola pregunta clara.\n\n"
        "Reglas estrictas:\n"
        "- Conserva EXACTAMENTE el mismo vocabulario y términos de la consulta original.\n"
        "- NO uses sinónimos ni reemplaces palabras.\n"
        "- NO agregues información adicional ni contexto extra.\n"
        f"- En IDIOMA: {_normalize_language_filter(language)}.\n"
        "- Devuelve SOLO la pregunta corregida, sin comillas ni texto introductorio.\n\n"
        "Pregunta corregida:"
    )
    try:
        rephrased = _invoke_chat(HAIKU_MODEL_ID, rephrase_prompt, max_tokens=80, temperature=0.0).strip() or question
    except Exception:
        rephrased = question

    print(f"PREGUNTA: ",question)
    print(f"REFRASEO: ",rephrased)
    mark(execution_times, 'rephrase_totalTime', t2)
    print(f"----------FIN - REFRASEO PREGUNTA----------")
    print("***********************************************************")
    t3 = tstart()
    print(f"----------INICIO - RETRIEVE DOCS----------")
    category = "Sin categoría"
    rephrased_chunks = []
    try:
        rephrased_chunks = retrieve_kb(rephrased, language)
        rephrased_chunks = normalize_results(rephrased_chunks, "rephrased")
        print(f"CHUNK REPHRASED")
        print("---------------")
    except Exception:
        rephrased_chunks = []

    mark(execution_times, 'categorizationAndRag', t3)
    print(f"----------FIN - RETRIEVE DOCS----------")
    print("***********************************************************")
    t4 = tstart()
    print(f"----------INICIO - SELECCIÓN ARTICULO----------")
    merged_docs = merge_results([], rephrased_chunks)
    candidate_docs = build_llm_candidates_payload(merged_docs, max_candidates=15)
    docs_text = ""
    for d in candidate_docs:
        raw_doc = (d.get("doc") or {}).get("raw", {})
        metadata = raw_doc.get("metadata") or {}

        description = metadata.get("context_short", "No hay 'context_short'")
        topics = metadata.get("context_bullets", "No hay 'context_bullets'")
        topics_str = str(topics)
        full_text = raw_doc.get("content", {}).get("text", "")
        not_covered = metadata.get("not_covered", "No hay 'not_covered'")
        intents = metadata.get("intents", "No hay 'intents'")

        urls_originales = metadata.get("urls", [])
        urls = [url for url in urls_originales if "lucid.app" in url]
        print("urls: ",urls)
        titulos_links = extraer_titulos_links(urls)

        if titulos_links:
            flujos_str = "\n".join(
                f"    {i}) {titulo}"
                for i, titulo in enumerate(titulos_links, start=1)
            )
        else:
            flujos_str = "    No hay flujos"

        docs_text += (
            f"Documento {d['id']}:\n"
            f"  -id: {d['id']}\n"
            f"  -titulo: {d['title']}\n"
            f"  -descripción: {description}\n"
            f"  -Flujos:\n{flujos_str}\n"
            f"  -topicos: {topics_str}\n"
            f"  -topicos no cubiertos (excluyente): {not_covered}\n"
            f"  -intents: {intents}\n"
        )

    print(docs_text)
    if not candidate_docs:
        selected_doc = {"title": "Título no encontrado", "raw": {}}
    else:
        prompt_chunks = (
            "Eres un sistema de selección de documentos y flujos relacionados (retrieval re-ranking).\n"
            "Tu tarea es elegir EXACTAMENTE 1 documento candidato que mejor responda la pregunta del usuario,\n"
            "y además elegir el o los flujos de ese documento que apoyan directamente la respuesta.\n\n"

            "REGLAS DE SALIDA (OBLIGATORIAS):\n"
            "1) Debes responder ÚNICAMENTE con este formato:\n"
            "   (<id_documento>)[<numeros_de_flujo_seleccionado>]\n"
            "2) El id del documento va entre paréntesis.\n"
            "3) Los números de los flujos seleccionados van entre corchetes.\n"
            "4) Si aplica un solo flujo, responde por ejemplo: (2)[1]\n"
            "5) Si aplican varios flujos, responde por ejemplo: (2)[1,2]\n"
            "6) No incluyas ningún texto adicional: sin explicación, sin comillas, sin markdown, sin prefijos, sin saltos extra.\n"
            "7) Debes elegir EXACTAMENTE 1 documento.\n"
            "8) Debes elegir solo flujos existentes dentro del documento seleccionado.\n"
            "9) Si el documento elegido no tiene ningún flujo claramente aplicable, responde con lista vacía, por ejemplo: (2)[]\n\n"

            "CÓMO DECIDIR EL DOCUMENTO:\n"
            "- Prioriza el documento que cubra MÁS directamente la pregunta del usuario.\n"
            "- Usa 'titulo', 'descripción', 'topicos' e 'intents' para entender el tema general y el objetivo del documento.\n"
            "- Usa 'Flujos' para identificar acciones, procesos o diagramas específicos que apoyan la respuesta.\n"
            "- Revisa 'topicos no cubiertos' para DESCARTAR documentos que explícitamente no traten partes clave de la pregunta.\n"
            "- No inventes información: decide SOLO con lo que aparece en los candidatos.\n\n"

            "CÓMO DECIDIR LOS FLUJOS:\n"
            "- Selecciona únicamente los flujos que estén directamente relacionados con la pregunta del usuario.\n"
            "- Si la pregunta pide una acción, proceso, procedimiento, paso a paso, flujo o cómo hacer algo, prioriza los flujos cuyo título coincida con esa intención.\n"
            "- Si varios flujos del mismo documento apoyan la pregunta, incluye todos sus números en orden ascendente.\n"
            "- Si un flujo es de un caso parecido pero no del caso preguntado, no lo selecciones.\n"
            "- No selecciones flujos de documentos no elegidos.\n"
            "- No selecciones un flujo solo porque existe; debe apoyar directamente la pregunta.\n\n"

            "HARD CONSTRAINTS (OBLIGATORIAS / PENALIZACIÓN MUY FUERTE SI NO SE CUMPLEN):\n"
            "0) Antes de aplicar coincidencias por escenario, identifica la INFORMACIÓN principal que el usuario está pidiendo, no solo el contexto mencionado. Expresiones como 'en caso de', 'derivado de', 'por', 'debido a' o 'cuando ocurre' suelen introducir contexto o causa. La intención principal de la pregunta pesa más que el contexto; por tanto, si el contexto coincide con un documento general pero la información pedida coincide con un documento más específico, elige el documento específico.\n"
            "1) Si la pregunta contiene un código/SSR o acrónimo específico, por ejemplo 'MAAS', 'INCU', 'WCHR',\n"
            "   SOLO son elegibles documentos donde ese código aparezca literalmente en titulo/descripción/topicos/flujos,\n"
            "   o un sinónimo directo inequívoco, por ejemplo MAAS = 'máxima asistencia'.\n"
            "2) Restricciones explícitas del enunciado: si la pregunta menciona canal, herramienta, sistema, país,\n"
            "   producto, normativa, mercado, rol o sigla operativa, esa mención debe aparecer o estar claramente implicada.\n"
            "   Para eventos amplios como atraso, cancelación, cambio involuntario o contingencia, trátalos como contexto,\n"
            "   salvo que la pregunta pida explícitamente el procedimiento general de ese evento.\n"
            "3) Si la pregunta exige un canal concreto, por ejemplo 'por email', penaliza fuerte documentos cuyo foco sea otro canal,\n"
            "   por ejemplo WhatsApp, Chat o voz, y que NO mencionen el canal solicitado.\n"
            "4) Si la pregunta incluye 2 o más condiciones, gana el documento que cumpla MÁS condiciones simultáneamente,\n"
            "   aunque otro documento sea muy fuerte en solo una de ellas.\n"
            "5) Si 'topicos no cubiertos' declara que NO cubre una restricción explícita, descártalo o penalízalo al máximo.\n"
            "6) Si la pregunta menciona explícitamente una normativa/mercado/rol o sigla operativa, por ejemplo\n"
            "   'Código de Consumo PE', 'PE/Perú', 'no show', 'LUA', 'HVC', SOLO son elegibles documentos que mencionen literalmente\n"
            "   esa misma señal en titulo/descripción/topicos/flujos/intents, o una variante obvia, por ejemplo 'Perú' para 'PE'.\n"
            "   Si ningún documento menciona esas señales, elige el más cercano por intención y con menor conflicto en 'topicos no cubiertos'.\n"
            "7) Si la pregunta contiene 'Staff travel' o 'consola', solo son elegibles docs que mencionen 'Staff travel'\n"
            "   o el sistema/consola correspondiente. Si ninguno lo menciona, recién ahí cae al más cercano.\n\n"

            "HEURÍSTICA DE SELECCIÓN:\n"
            "A) Primero elige el documento más alineado con la pregunta.\n"
            "B) Luego, dentro de ese documento, elige los flujos que apoyen directamente la pregunta.\n"
            "C) Si el título de un flujo coincide semánticamente con la pregunta, es fuerte candidato.\n"
            "D) Si varios documentos parecen relevantes, elige el que tenga mejor combinación de descripción + topicos + intents + flujos.\n"
            "E) Si ninguno responde perfectamente, elige el MÁS cercano con mejor cobertura parcial y menor conflicto.\n\n"

            "EJEMPLOS DE SALIDA:\n"
            "- Pregunta: '¿como es la accion o flujo de exceso en equipaje?'\n"
            "- Documento elegido: 2\n"
            "- Flujos relevantes dentro del documento:\n"
            "  1) Flujo Emisión Exceso de Equipaje\n"
            "  2) Flujo Emisión Oversize de Equipaje\n"
            "- Respuesta correcta si solo aplica exceso de equipaje: (2)[1]\n"
            "- Respuesta correcta si aplican exceso y oversize: (2)[1,2]\n\n"

            "IMPORTANTE:\n"
            "- No inventes IDs.\n"
            "- No inventes números de flujo.\n"
            "- No expliques tu decisión.\n"
            "- No devuelvas títulos de flujos, solo sus números.\n"
            "- La númeración de flujos se reinicia (vuelve a 1) por cada documento"
            "- Debes escoger un único documento incluso si la cobertura es parcial.\n\n"

            f"PREGUNTA DEL USUARIO:\n{question}\n\n"

            "DOCUMENTOS CANDIDATOS (formato fijo):\n"
            "Cada candidato aparece como:\n"
            "Documento X:\n"
            "  -id: <id>\n"
            "  -titulo: <titulo>\n"
            "  -descripción: <descripción>\n"
            "  -Flujos:\n"
            "    1) <titulo_flujo_1>\n"
            "    2) <titulo_flujo_2>\n"
            "    3) <titulo_flujo_3>\n"
            "  -topicos: <topicos>\n"
            "  -topicos no cubiertos (excluyente): <not_covered>\n"
            "  -intents: <intents>\n\n"

            f"{docs_text}\n\n"

            "RESPUESTA OBLIGATORIA, SOLO EN FORMATO (<id_documento>)[<numeros_de_flujo_seleccionado>]:"
        )
        respuesta_ia = _invoke_chat(SONNET_45_MODEL_ID, prompt_chunks, max_tokens=100, temperature=0.0).strip()
    
        chosen_id, chosen_flujos = parsear_respuesta_ia(respuesta_ia)

        print(prompt_chunks)

        print("CHOSEN DOC ID - IA:", chosen_id)
        print("CHOSEN FLUJOS - IA:", chosen_flujos)
    
        chosen_candidate = next((c for c in candidate_docs if c["id"] == chosen_id), None)

        if not chosen_candidate:
            print("No se encontró candidato para ese id, fallback a primero")
            chosen_candidate = candidate_docs[0]
            
        selected_doc = chosen_candidate["doc"]

    print("---------")
    print(f"ARTICULO A UTILIZAR:")
    print(f"{selected_doc}")
    print("---------")

    mark(execution_times, 'chunkSelection_totalTime', t4)
    print(f"----------FIN - SELECCIÓN ARTICULO----------")
    print("***********************************************************")
    t5 = tstart()
    print(f"----------INICIO - INYECCIÓN DEPENDENCIAS----------")

    title_to_show = selected_doc.get("title", "Título no encontrado")
    print("Title to show:", title_to_show)

    raw = selected_doc.get("raw", {})  # chunk original de Bedrock

    rag_files = (
        raw.get('location', {})
        .get('s3Location', {})
        .get('uri', '')
        .replace(CONFIG.s3_files_prefix, '')
    )

    sel_text = (
        raw.get('content', {})
        .get('text', '')
    )

    meta = raw.get('metadata', {})

    print("Sel_text original:")
    print(sel_text)


    # =====================================================
    # RESCATAR FLUJOS SELECCIONADOS POR LA IA
    # =====================================================

    selected_flows = obtener_flujos_seleccionados(meta, chosen_flujos)

    print("Flujos seleccionados metadata:")
    print(selected_flows)

    selected_flow_urls_norm = {
        flujo["url_norm"]
        for flujo in selected_flows
        if flujo.get("url_norm")
    }

    selected_flow_titles_norm = {
        flujo.get("titulo_norm") or normalizar_texto_para_match(flujo.get("titulo", ""))
        for flujo in selected_flows
        if flujo.get("titulo_norm") or flujo.get("titulo")
    }

    print("Selected flow URLs norm:")
    print(selected_flow_urls_norm)

    print("Selected flow titles norm:")
    print(selected_flow_titles_norm)


    # =====================================================
    # LINKS HTML SOLO DE FLUJOS SELECCIONADOS
    # =====================================================

    links_seleccionados = [
        f"{flujo['url']};{flujo['titulo']}"
        for flujo in selected_flows
    ]

    print("Links seleccionados raw:")
    print(links_seleccionados)

    links = convertir_links_a_html(links_seleccionados)

    print("Links seleccionados HTML:")
    print(links)


    # =====================================================
    # METADATA RESPONSE
    # =====================================================

    metadata_response = {
        "tablas": meta.get("tablas", []),
        "flujos": meta.get("flujos", []),
    }


    # =====================================================
    # GLOSARIO
    # =====================================================

    glosario_uris = meta.get('glosario', []) or []

    glosario_str = ""

    for uri in glosario_uris:
        try:
            raw_txt, used_uri = fetch_with_fallback(uri)
            txt = normalize_text(raw_txt)
            glosario_str += txt + "\n"

        except Exception:
            glosario_str += f"No encontre contenido en glosario: {uri}\n"
            continue


    # =====================================================
    # FASE 1: INYECTAR SOLO TABLAS
    # =====================================================

    print("----------FASE 1 - INYECCIÓN TABLAS----------")

    tablas_uris = meta.get('tablas', []) or []

    print("TABLAS URIS desde metadata:")
    print(tablas_uris)

    metadata_tablas, uri_to_content_tablas = cargar_dependencias(tablas_uris)

    print("Metadata tablas:")
    print(metadata_tablas)

    context_text = inyectar_dependencias_por_tipo(
        texto=sel_text,
        uri_to_content=uri_to_content_tablas,
        tipo="TABLA"
    )

    print("Texto después de inyectar TABLAS:")
    print(context_text)


    # =====================================================
    # FASE 2: FILTRAR FLUJOS NO SELECCIONADOS
    # SOBRE EL TEXTO RESULTANTE DE TABLAS
    # =====================================================

    print("----------FASE 2 - FILTRO FLUJOS SELECCIONADOS----------")

    print("Texto ANTES de filtrar FLUJOS:")
    print(context_text)

    context_text = filtrar_flujos_no_seleccionados_en_texto(
        sel_text=context_text,
        selected_flow_urls_norm=selected_flow_urls_norm,
        selected_flow_titles_norm=selected_flow_titles_norm
    )

    print("Texto DESPUÉS de filtrar FLUJOS no seleccionados:")
    print(context_text)


    # =====================================================
    # FASE 3: EXTRAER FLUJOS DESPUÉS DE INYECTAR TABLAS
    # =====================================================

    print("----------FASE 3 - EXTRACCIÓN FLUJOS POST-TABLAS----------")

    flujos_uris = extraer_uris_por_tipo(
        texto=context_text,
        tipo="FLUJO"
    )

    print("FLUJOS URIS encontrados después de inyectar tablas y filtrar:")
    print(flujos_uris)


    # =====================================================
    # FASE 4: INYECTAR SOLO FLUJOS SELECCIONADOS
    # =====================================================

    print("----------FASE 4 - INYECCIÓN FLUJOS----------")

    metadata_flujos, uri_to_content_flujos = cargar_dependencias(flujos_uris)

    print("Metadata flujos:")
    print(metadata_flujos)

    context_text = inyectar_dependencias_por_tipo(
        texto=context_text,
        uri_to_content=uri_to_content_flujos,
        tipo="FLUJO"
    )

    print("Texto final después de inyectar FLUJOS:")
    print(context_text)


    # =====================================================
    # METADATA LIST FINAL
    # =====================================================

    metadata_list = metadata_tablas + metadata_flujos

    uri_to_content: Dict[str, str] = {}

    for item in metadata_list:
        uri_norm = normalize_s3_uri(item.get("uri", ""))
        content = item.get("content", "") or ""
        uri_to_content[uri_norm] = content

    print("Metadata List final:")
    print(metadata_list)


    # =====================================================
    # CONTEXTO FINAL
    # =====================================================

    context_text = "Glosario: " + glosario_str + "Artículo: " + context_text

    print("Context text final:")
    print(context_text)
    # print(f"----------INICIO - INYECCIÓN DEPENDENCIAS----------")

    mark(execution_times, 'loadChunk', t5)
    print(f"----------FIN - INYECCIÓN DEPENDENCIAS----------")
    print("***********************************************************")
    t6 = tstart()
    print(f"----------INICIO - GENERAR RESPUESTA----------")
    articulo = context_text
    
    print("LEN ARTICULO: ",len(articulo))
    print(articulo)

    answer_prompt = (
        "STRICT INSTRUCTIONS (FOLLOW EXACTLY):\n"
        "\n"
        "1) SINGLE SOURCE OF TRUTH:\n"
        "   — You may only use the INFORMATION section.\n"
        "   — Do not use external knowledge.\n"
        "   — Do not make inferences beyond the text.\n"
        "   — Do not invent steps, deadlines, policies, processes, or conditions that are not mentioned.\n"
        "   — FORBIDDEN: Do not invent exceptions, do not assume markets, and do not name countries that are not explicitly written in the provided INFORMATION.\n"
        "\n"
        "2) COMPLETE COVERAGE:\n"
        "   — If the INFORMATION shows different rules by country (Argentina, Chile, Colombia, Uruguay, etc.), "
        "you must include ALL explicit rules for EACH mentioned country.\n"
        "   — Do not assume the question refers only to Chile.\n"
        "\n"
        "3) WHEN INFORMATION IS MISSING:\n"
        "   — If there is NOT enough evidence in INFORMATION to answer the question, respond with exactly this string and nothing else: NOT FOUND\n"
        "   — Do not wrap NOT FOUND in <RESUMEN>, [ADICIONAL], quotes, bullets, explanations, or any other text.\n"
        "   — Do not translate NOT FOUND, regardless of the selected response language.\n"
        "   — If the information is PARTIAL, do NOT use NOT FOUND. Answer only what is explicitly shown and clearly mention which points are not documented.\n"
        "\n"
        "4) RESPONSE STYLE:\n"
        f"   — Respond only in LANGUAGE: {_normalize_language_filter(language)}.\n"
        "   — The language value will be either Spanish or Portuguese.\n"
        "   — Do not mix languages, except for proper names, system names, labels, or terms that must remain unchanged.\n"
        "   — Always keep these terms exactly as written, without translating them: Latam Wallet, RESUMEN, ADICIONAL, NOT FOUND.\n"
        "   — First, answer the question DIRECTLY.\n"
        "   — COMPLETE FLOW RULE: If the question or process involves a support flow or operational steps, you MUST detail the procedure from beginning to end, but only with steps explicitly documented in INFORMATION. Include final actions in the systems, settlement, and the final closure or notification to the customer only if those elements are documented. If part of the flow is missing, clearly state which part is not documented.\n"
        "   — LOGICAL CONDITIONALS RULE: Pay strict attention to the conditions in the question (e.g., 'if it was completed successfully', 'if it failed', 'if the person is a minor'). You must search the INFORMATION for the process branch that corresponds EXACTLY to that state. Do not mix steps from a successful process with steps from a failed process or an exception process.\n"
        "   — ANCHORING DISCARD RULE: Do not choose a section of the INFORMATION only because it shares identical words with the question (e.g., 'Latam Wallet'). If the question describes an issue with one tool, but the documented solution requires migrating to another one (e.g., moving from Wallet to Bank Transfer in TV Web), you must prioritize the procedure that solves the specific situation described, not the one that repeats the words from the question more often.\n"
        "   — Only include links if they appear explicitly in INFORMATION, start with https://, and belong to an allowed domain such as drive.google.com or docs.google.com.\n"
        "   — Use terminology faithful to the document, but paraphrase whenever possible. Do not copy full passages literally.\n"
        "\n"
        "IMPORTANT OUTPUT EXCEPTION:\n"
        "   — The RESUMEN and ADICIONAL format applies only when there is enough information to answer at least partially.\n"
        "   — If there is not enough evidence in INFORMATION, the entire response must be exactly: NOT FOUND\n"
        "\n"
        "--------------------------------------------------\n"
        "-INFORMATION:\n"
        f"{articulo}.\n"
        "\n"
        "--------------------------------------------------\n"
        "-AGENT QUESTION:\n"
        f"{question}\n"
        "--------------------------------------------------\n"
        "5) OUTPUT FORMAT: RESUMEN (STRICT):\n"
        "   <RESUMEN>\n"
        "   — A single direct and specific paragraph answering ONLY the question.\n"
        "   — Include EXACT numbers, EXACT conditions, and wording faithful to the document.\n"
        "   — It must open with '<RESUMEN>' and close with '</RESUMEN>'.\n"
        "   </RESUMEN>\n"
        "\n"
        "5.1) OUTPUT FORMAT: ADICIONAL (STRICT):\n"
        "   [ADICIONAL]\n"
        "   — List 2 to 10 bullet points with relevant additional details, without inventing anything.\n"
        "   — Include country-specific rules if they exist.\n"
        "   — Do not add steps that are not mentioned in INFORMATION.\n"
        "   — It must open with '[ADICIONAL]' and close with '[/ADICIONAL]'.\n"
        "   [/ADICIONAL]\n"
        "\n"
    )
    try:
        MAX_RETRIES = 7

        full_answer = None
        timed_out = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                full_answer, timed_out = _invoke_chat_with_timeout(
                    GPT_OSS_120_MODEL_ID,
                    answer_prompt,
                    max_tokens=ANSWER_MAX_TOKENS,
                    temperature=0.0
                )

                full_answer = (full_answer or "").strip()

                # Si el modelo falló (según tu contrato), reintentamos
                if full_answer == "Intermitencia de Servicio":
                    if attempt < MAX_RETRIES:
                        continue
                    # último intento y falló -> rompemos para caer al fallback
                    break

                # OK: limpiamos y salimos
                # full_answer = remove_reasoning_blocks(full_answer)
                full_answer, reasoning_blocks = split_reasoning_blocks(full_answer)
                reasoning_text = "\n\n".join(reasoning_blocks)
                print("Razonamiento: ",reasoning_text)
                break

            except Exception:
                # Excepción inesperada: reintenta también (opcional)
                if attempt < MAX_RETRIES:
                    continue
                full_answer = "Intermitencia de Servicio"
                break

        # Fallback final si quedó en error / None / vacío por cualquier motivo
        if not full_answer or full_answer == "Intermitencia de Servicio":
            full_answer = "Intermitencia de Servicio"

        print("FULL ANSWER:",full_answer)
    except Exception:
        full_answer = "Intermitencia de Servicio"

    print("len prompt qna:",len(answer_prompt))
    print("prompt qna:",answer_prompt)

    mark(execution_times, 'generateAnswer_totalTime', t6)
    print(f"----------FIN - GENERAR RESPUESTA----------")
    print("***********************************************************")
    if not timed_out:
        t7 = tstart()
        print("----------INICIO - PARSEAR RESUMEN Y INFO ADICIONAL----------")
        resumen = ""
        info_adicional = ""

        # Normalización ligera de saltos HTML
        full_answer_norm = re.sub(r'(?i)<br\s*/?>', '\n', full_answer or "")

        # Patrones reutilizables
        RESUMEN_RE = r'(?:RESUMEN|RESUMÉN|RESUMN|RESUME)'
        ADICIONAL_RE = r'(?:informaci[oó]n\s+adicional|informaci[oó]n\s+adicial|adicional|adicial|detalles\s+adicionales|detalles\s+adiciales|additional)'

        # ======== RESUMEN ========
        m = re.search(
            rf'''
            \[\s*{RESUMEN_RE}\s*\]\s*
            (.*?)                                                     # contenido
            \s*
            (?=
                \[
                /\s*{RESUMEN_RE}\s*
                \]
                |
                \[\s*{ADICIONAL_RE}\s*\]
                |
                $
            )
            ''',
            full_answer_norm, flags=re.IGNORECASE | re.DOTALL | re.UNICODE | re.VERBOSE
        )
        if m:
            resumen = m.group(1).strip()
        else:
            m = re.search(
                rf'''
                <\s*{RESUMEN_RE}\s*>\s*
                (.*?)                                                 # contenido
                \s*
                (?=
                    </\s*{RESUMEN_RE}\s*>
                    |
                    </\s*>
                    |
                    \[\s*{ADICIONAL_RE}\s*\]
                    |
                    $
                )
                ''',
                full_answer_norm, flags=re.IGNORECASE | re.DOTALL | re.UNICODE | re.VERBOSE
            )
            if m:
                resumen = m.group(1).strip()
            else:
                m = re.search(r'<\s*([^<>]+?)\s*>', full_answer_norm, flags=re.DOTALL)
                if m:
                    inner = m.group(1).strip()
                    if inner.upper() not in ("RESUMEN", "RESUMÉN", "RESUMN", "RESUME"):
                        resumen = inner

        # ======== INFO_ADICIONAL ========
        m = re.search(
            rf'''
            \[\s*{ADICIONAL_RE}\s*\]\s*:?\s*
            (.*?)                                    # contenido
            \s*
            \[\s*/\s*{ADICIONAL_RE}\s*\]
            ''',
            full_answer_norm, flags=re.IGNORECASE | re.DOTALL | re.UNICODE | re.VERBOSE
        )
        if m:
            info_adicional = m.group(1).strip()
        else:
            m = re.search(
                rf'''
                \[\s*{ADICIONAL_RE}\s*\]\s*:?\s*
                (.*?)                                 # contenido
                \s*
                (?=
                    </\s*{ADICIONAL_RE}\s*>
                    | \[/\s*{ADICIONAL_RE}\s*\]
                    | $
                )
                ''',
                full_answer_norm, flags=re.IGNORECASE | re.DOTALL | re.UNICODE | re.VERBOSE
            )
            if m:
                info_adicional = m.group(1).strip()
            else:
                m2 = re.search(r'\[\s*([^\[\]]+?)\s*\]', full_answer_norm, flags=re.DOTALL | re.UNICODE)
                if m2:
                    candidate = m2.group(1).strip()
                    if not re.fullmatch(
                        ADICIONAL_RE,
                        candidate,
                        flags=re.IGNORECASE
                    ):
                        info_adicional = candidate

        # ======== RECUPERACIÓN PARA CASO MALFORMADO: <RESUMEN> ... [/ADICIONAL] / [/ADDITIONAL] ... ========
        try:
            closing_add_re = rf'\[/\s*{ADICIONAL_RE}\s*\]'
            opening_res_re = rf'<\s*{RESUMEN_RE}\s*>'

            if not info_adicional:
                text = full_answer_norm or ""

                # Caso A: el RESUMEN quedó con un cierre [/ADICIONAL] o [/ADDITIONAL] dentro
                if resumen and re.search(closing_add_re, resumen, flags=re.IGNORECASE):
                    parts = re.split(closing_add_re, resumen, maxsplit=1, flags=re.IGNORECASE)
                    resumen = (parts[0] or "").strip()
                    tail = (parts[1] if len(parts) > 1 else "").strip()

                    tail = re.sub(closing_add_re + r'\s*$', "", tail, flags=re.IGNORECASE).strip()

                    if tail:
                        info_adicional = tail

                # Caso B: existe <RESUMEN> y uno o más cierres [/ADICIONAL] o [/ADDITIONAL] sin apertura
                if not info_adicional and re.search(opening_res_re, text, flags=re.IGNORECASE) and re.search(closing_add_re, text, flags=re.IGNORECASE):
                    after_resumen = re.split(opening_res_re, text, maxsplit=1, flags=re.IGNORECASE)[1]
                    before_close, *rest = re.split(closing_add_re, after_resumen, maxsplit=1, flags=re.IGNORECASE)
                    cand_resumen = (before_close or "").strip()
                    cand_tail = (rest[0] if rest else "").strip()

                    cand_tail = re.sub(closing_add_re + r'\s*$', "", cand_tail, flags=re.IGNORECASE).strip()

                    if cand_resumen and (not resumen or resumen.upper() in ("RESUMEN", "RESUMÉN", "RESUMN", "RESUME")):
                        resumen = cand_resumen
                    if cand_tail:
                        info_adicional = cand_tail

            # Limpiezas finales menores: quitar prefijos como "FULL ANSWER:" si quedaron
            def _clean_prefix(s: str) -> str:
                s2 = s.strip()
                s2 = re.sub(r'^\s*FULL\s*ANSWER\s*:\s*', '', s2, flags=re.IGNORECASE)
                return s2.strip()

            resumen = _clean_prefix(resumen or "")
            info_adicional = _clean_prefix(info_adicional or "")

        except Exception:
            # Falla silenciosa
            pass
        # print(f"----------INICIO - PARSEAR RESUMEN Y INFO ADICIONAL----------")

        # Limpieza Resumen
        resumen = clean_response_text(resumen)
        resumen = remove_resumen_tag(resumen)
        if resumen is not None and resumen != "":
            resumen = format_resumen(resumen)
            resumen = convertir_links_html(resumen)
            resumen = formatear_texto(resumen)
            if _normalize_language_filter(language) == 'portugues':
                resumen += '<br><br> <b>Fonte:</b> '+title_to_show
            else:
                resumen += '<br><br> <b>Fuente:</b> '+title_to_show
        else:
            resumen = "No encontrado en Información"
            
        print("*****************RESUMEN*************************")
        print(resumen)
        print("*****************FIN RESUMEN*************************")
        # Limpieza Info Adicional para Frontal
        print("*****************INFO ADICIONAL*************************")
        print(info_adicional)
        print("*****************FIN INFO ADICIONAL*************************")
        raw_additional = format_to_html(info_adicional)
        raw_additional = clean_response_text(raw_additional)

        # Limpieza Info Adicional para DynamoDB
        raw_additional_clean = clean_response_text(info_adicional)

        mark(execution_times, 'parseAnswer', t7)
        print(f"----------FIN - PARSEAR RESUMEN Y INFO ADICIONAL----------")
        t8 = tstart()
        print("*************************************************************")
        print(f"----------INICIO - GUARDAR HISTORIAL----------")
        now_santiago = datetime.now(ZoneInfo("America/Santiago")).isoformat()
        total_time = round(sum(execution_times.values()), 2)
        question_item = {
            'id': str(uuid.uuid4()),
            'session_id': session_id or '',
            'agent_name': name or 'No encontré Nombre',
            'agent_email': email or 'No encontré Email',
            'question': question or '',
            'rephrased': rephrased or '',
            'category': category or 'Sin categoría',
            'answer': resumen or 'ERROR_GENERATION',
            'additional_information': raw_additional_clean or 'ERROR_GENERATION',
            'rag_files': rag_files or '',
            'date': now_santiago or '',
            'total_execution_time': f"{total_time:.2f}",
            'fragments': articulo,
            'language': _normalize_language_filter(language),
        }
        try:
            questions_table.put_item(Item=question_item)
        except ClientError as e:
            print(f"Error saving question: {e}")
        mark(execution_times, 'saveQuestion', t8)
        print(f"----------FIN - GUARDAR PREGUNTA----------")
        print("*************************************************************")
        t9 = tstart()
        print(f"----------INICIO - ACTUALIZAR HISTORIAL SESION ----------")
        new_history = history + [question]
        conv_item = {
            'user_id': session_id,
            'history': new_history,
            'last_updated': now_santiago
        }
        try:
            conversations_table.put_item(Item=conv_item)
        except ClientError as e:
            print(f"Error updating history: {e}")
        
        mark(execution_times, 'saveHistory', t9)
        print(f"-------------FIN - ACTUALIZAR HISTORIAL SESION -------------")
        print("*************************************************************")
        t10 = tstart()
        print(f"----------INICIO - BUILD RESPONSE ----------")
        error_msg = "No encontrado en Información"
        if not resumen.strip(): 
            resumen = error_msg
        if not raw_additional.strip() or raw_additional.strip() == '[]':
            raw_additional = error_msg
            raw_additional_clean = error_msg
        if resumen == 'No encontrado en Información':
            raw_additional_clean = 'No encontrado en Información'

        response_body = {
            "sessionAttributes": {
                "client_transcript": question,
                "rephrased_message": rephrased,
                "kbid": KB_ID,
                "category": category or 'Sin categoría',
                "raw_message": resumen.replace("\n", "<br>"),
                "raw_additional": raw_additional or 'Intermitencia de Servicio',
                "raw_additional_clean": raw_additional_clean.replace("\n", "") or 'Intermitencia de Servicio',
                "fragments_used": articulo,
                "session_id": session_id,
                "rag_files": rag_files,
                "links": links,
                "metadata": metadata_response,
                "language": _normalize_language_filter(language),
            },
            "total_execution_time": total_time
        }
        if INCLUDE_EXEC_TIMES:
            response_body["execution_times"] = execution_times

        mark(execution_times, "buildResponse", t10)

    else:
        resumen = full_answer  # por ejemplo
        raw_additional = full_answer
        raw_additional_clean = full_answer
        category = category or 'Sin categoría'

        t10 = tstart()  # o time.time(), como estés midiendo
        total_time = round(sum(execution_times.values()), 2)
        mark(execution_times, "buildResponse", t10)

        response_body = { 
            "sessionAttributes": {
                "client_transcript": question,
                "rephrased_message": rephrased,
                "kbid": KB_ID,
                "category": category,
                "raw_message": resumen.replace("\n", "<br>") if resumen else 'No encontrado en Información.',
                "raw_additional": raw_additional or 'ERROR_GENERATION',
                "raw_additional_clean": raw_additional_clean.replace("\n", "") if raw_additional_clean else 'No encontrado en Información.',
                "fragments_used": context_text,
                "session_id": session_id,
                "rag_files": rag_files,
                "links": links,
                "metadata": metadata_list,
                "language": _normalize_language_filter(language),
            },
            "total_execution_time": total_time
        }
        if INCLUDE_EXEC_TIMES:
            response_body["execution_times"] = execution_times

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(response_body)
    }