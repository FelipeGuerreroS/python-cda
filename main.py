import os
import re
import json
import time
import threading
import base64
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import boto3
import requests
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
import concurrent.futures
from urllib.parse import urlparse, urlunparse
from typing import Any, Mapping, Tuple, Optional
from decimal import Decimal
from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError, ReadTimeoutError, ConnectTimeoutError
from time import perf_counter
from collections import defaultdict

## FUNCIONES PARA MARCAR TIEMPOS DE EJECUCIÓN

def tstart():
    return perf_counter()

def mark(execution_times: dict, key: str, t0: float):
    execution_times[key] = round(perf_counter() - t0, 4)

## VARIABLES GLOBALES

# ENV
AWS_REGION           = os.environ['REGION']
CONV_TABLE_NAME      = os.environ['CONVERSATIONS_TABLE']
QUESTIONS_TABLE_NAME = os.environ['QUESTIONS_TABLE']
KB_ID                = os.environ['KB_ID']

# MODELOS
HAIKU_MODEL_ID       = os.environ['HAIKU_MODEL_ID']
SONNET_MODEL_ID      = os.environ['SONNET_MODEL_ID']  
SONNET_37_MODEL_ID   = os.environ['SONNET_37_MODEL_ID']
SONNET_4_MODEL_ID    = os.environ['SONNET_4_MODEL_ID']
TITAN_PREMIER_MODEL_ID = os.environ['TITAN_PREMIER_MODEL_ID']
TITAN_EXPRESS_MODEL_ID = os.environ['TITAN_EXPRESS_MODEL_ID']
GPT_OSS_120_MODEL_ID   = os.environ['GPT_OSS_120_MODEL_ID']

#OUTPUT IA
ANSWER_MAX_TOKENS = int(os.environ.get('ANSWER_MAX_TOKENS', '6000'))
GUARDRAIL_ID = "sdsuv2s4z9op"
GUARDRAIL_VER = "DRAFT"
TRACE = "ENABLED"
ANTHROPIC_VERSION    = "bedrock-2023-05-31"

#OTROS
CATEGORIES = os.environ.get('CATEGORIES', '').split(',') #DEPRECADO: SE DEJA POR SI CAMBIA A FUTURO
INCLUDE_EXEC_TIMES = os.environ.get('INCLUDE_EXEC_TIMES', 'true').lower() in ('true', '1', 'yes')

## CLIENTES AWS
dynamo              = boto3.resource('dynamodb', region_name=AWS_REGION)
bedrock_agent       = boto3.client('bedrock-agent-runtime', region_name=AWS_REGION)
bedrock_runtime     = boto3.client('bedrock-runtime', region_name=AWS_REGION)
bedrock_runtime_usw2= boto3.client('bedrock-runtime', region_name='us-west-2') 
s3_client           = boto3.client('s3', region_name=AWS_REGION)

#TABLAS DYNAMO
conversations_table = dynamo.Table(CONV_TABLE_NAME)
questions_table     = dynamo.Table(QUESTIONS_TABLE_NAME)

## MENSAJES PROTOTIPOS
error_msg = "Error"

_S3_RE = re.compile(
    r"""
    s3://[^\s'\"?#]+/        # prefijo y carpetas
    (?P<basename>[^/]+\.(?P<ext>csv|json))   # archivo + extensión
    (?:[?#][^\s'"]*)?        # opcional query/anchor
    """,
    re.IGNORECASE | re.VERBOSE
)

def parse_request(event):
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
        return bedrock_runtime_usw2
    return bedrock_runtime

def _invoke_chat(model_id, prompt_text, max_tokens, temperature, top_p=None, stop_sequences=None):
    # DIFERENCIACIÓN POR MODELO (Anthropic Claude / Amazon Titan / OpenAI OSS)
    is_anthropic = model_id.startswith("us.anthropic.")
    is_titan     = model_id.startswith("amazon.titan-text")
    is_openai    = model_id.startswith("openai.gpt-oss")

    accept       = "application/json"
    content_type = "application/json"

    system_prompt = "Realiza lo que se te pida lo más profesional posible."

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

def _invoke_chat_with_timeout(model_id, prompt_text, max_tokens, temperature, top_p=None, stop_sequences=None, timeout_seconds=26):
    """
    Envuelve _invoke_chat con un timeout duro.
    - Si el modelo responde antes del timeout → devuelve (respuesta, False)
    - Si se pasa del timeout → devuelve ("Intermitencia de Servicio", True)
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

def retrieve_kb(txt):
    kb_resp = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 6}},
        retrievalQuery={"text": txt}
    )
    return kb_resp.get('retrievalResults', [])

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
        if item.get("description") is not None:
            d["description"] = item["description"]
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
        if item.get("description") is not None and d["description"] is None:
            d["description"] = item["description"]
        if item.get("idioma") is not None and d["idioma"] is None:
            d["idioma"] = item["idioma"]
        d["from_rephrased"] = {
            "rank": idx + 1,
            "score": item["score"]
        }
        if d["doc"] is None:
            d["doc"] = item

    return list(merged.values())

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

def build_llm_candidates_payload(merged_docs, max_candidates=5):
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

def normalizar_links(lista_links):
    """
    Recibe una lista de URLs y devuelve una lista sin duplicados,
    normalizando pequeñas variaciones (barra final, comillas, etc.)
    """
    urls_limpias = set()

    for link in lista_links:
        if not isinstance(link, str):
            continue

        link = link.strip().strip('"').strip("'")

        parsed = urlparse(link)

        netloc = parsed.netloc.lower()

        path = parsed.path.rstrip('/') if parsed.path != '/' else parsed.path

        normalized = urlunparse((parsed.scheme, netloc, path, '', parsed.query, parsed.fragment))

        urls_limpias.add(normalized)

    # Convertir de nuevo a lista ordenada
    return sorted(urls_limpias)

def normalize_s3_uri(u: str) -> str:
    """Normaliza variantes comunes de prefijo s3 y quita puntuación final habitual."""
    if not isinstance(u, str):
        return ""
    u = u.strip()
    u = re.sub(r'^s3(?:::|//)', 's3://', u)
    u = re.sub(r'^s3:/', 's3://', u)
    u = u.rstrip('),.;')
    return u

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

def remove_resumen_tag(text: str) -> str:
    return re.sub(r'\[\s*/\s*RESUMEN\s*\]', '', text, flags=re.IGNORECASE)

def clean_newlines(text):
    if text:
        return text.replace("\\n", "\n").replace("\n\n", "\n\n").strip()
    return ""

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
    resumen = resumen.replace(". ", ".\n")
    resumen = resumen.replace("\n", "<br><br>")
    resumen_formateado = f"<p>{resumen}</p>"
    
    return resumen_formateado

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

def lambda_handler(event, context):
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
        + "\n\nReformula en una sola pregunta la consulta anterior. Devuelve SOLO la pregunta.\n\n"
        "Pregunta reformulada:"
    )
    try:
        rephrased = _invoke_chat(HAIKU_MODEL_ID, rephrase_prompt, max_tokens=90, temperature=0.0).strip() or question
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
    refs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fut_kb = executor.submit(retrieve_kb, question)
        try:
            refs = fut_kb.result()
        except Exception:
            refs = []
    mark(execution_times, 'categorizationAndRag', t3)

    t4 = tstart()
    # 5) Chunk selection
    top_refs = refs[:3]

    fragments = []          # aquí guardaremos los 2 artículos ya “expandido” con TABLA/FLUJO resueltas
    all_links = []          # si quieres tener todos los links juntos
    rag_files = []          # por si quieres saber de qué archivo salió cada fragmento

    # --- Regex y helpers comunes para todos los resultados ---
    pattern = re.compile(r'\[(TABLA)\]\s*(s3://\S+)', flags=re.IGNORECASE)

    for rank, selected in enumerate(top_refs):
        # --- Info básica del resultado ---
        rag_file = (
            selected
            .get('location', {})
            .get('s3Location', {})
            .get('uri', '')
            .replace('s3://files-cda/txt/Artículos/', '')
        )
        rag_files.append(rag_file)

        sel_text = selected.get('content', {}).get('text', '').replace('\r', '')
        print(f"Resultado #{rank+1} - sel_text:", sel_text)

        # --- Metadatos: tablas + flujos + urls ---
        uris = (
            selected.get('metadata', {}).get('tablas', [])
        )

        links = selected.get('metadata', {}).get('urls', [])
        links = normalizar_links(links)
        all_links.extend(links)

        print(f"Resultado #{rank+1} - LINKS:", links)

        extra_sections = []
        metadata_list = []

        for uri in uris:
            try:
                raw_txt = fetch_s3_text(uri)
                txt = re.sub(r'\s+', ' ', raw_txt.replace('\n', '')).strip()
                metadata_list.append({"uri": uri, "content": txt})
            except Exception as e:
                error_msg = f"Error {e}"
                extra_sections.append(f"{uri}: {error_msg}")
                metadata_list.append({"uri": uri, "content": error_msg})

        # --- Construye un mapa uri -> contenido (URIs normalizadas) ---
        uri_to_content = {
            normalize_s3_uri(item.get("uri", "")): (item.get("content", "") or "")
            for item in metadata_list
        }

        def replace_match(m: re.Match) -> str:
            kind = m.group(1).upper()  # TABLA o FLUJO
            uri_raw = m.group(2)
            uri_norm = normalize_s3_uri(uri_raw)
            content = uri_to_content.get(uri_norm, "")
            file_name = os.path.basename(uri_norm) or uri_norm.split('/')[-1]

            if not content:
                # Si no lo encontramos, conserva la referencia original y marca el fallo
                return f'[{kind}] {uri_raw} [NO ENCONTRADO EN METADATA]'

            fin_tag = end_tag_for_uri(uri_norm, kind)
            return f'[{kind}] {{ {content} }} [{fin_tag}]'

        # --- Aplica el reemplazo al texto del artículo actual ---
        sel_text_replaced = pattern.sub(replace_match, sel_text)

        # Guardamos cada resultado como un fragmento independiente
        fragments.append({
            "rank": rank,             # 0 = mejor, 1 = segundo mejor
            "rag_file": rag_file,     # nombre del archivo base
            "text": sel_text_replaced,  # texto del artículo con TABLA/FLUJO ya imputados
            "links": links,           # urls asociadas a este resultado
            "metadata": metadata_list # lista de tablas/flujos resueltas
        })

    # Si quieres tener un solo texto concatenado de los dos resultados:
    texto_unico = "\n\n---\n\n".join(frag["text"] for frag in fragments)

    
    print(f"----------FIN - RETRIEVE DOCS----------")
    print("***********************************************************")
  

    mark(execution_times, 'chunkSelection_totalTime', t4)
    # print(f"----------FIN - SELECCIÓN ARTICULO----------")
    # print("***********************************************************")
    t5 = tstart()


    mark(execution_times, 'loadChunk', t5)
    print(f"----------FIN - INYECCIÓN DEPENDENCIAS----------")
    print("***********************************************************")
    t6 = tstart()
    print(f"----------INICIO - GENERAR RESPUESTA----------")
    articulo = "\n\n".join(texto_unico)
    print("LEN ARTICULO: ",len(articulo))
    if len(articulo) > 31000:
        print("articulo cortado")
        articulo = articulo[:31000]
    else:
        articulo = articulo

    answer_prompt = (
        "INSTRUCCIONES ESTRICTAS:\n"
        "\n"
        "1) Tu ÚNICA fuente es la sección INFORMACIÓN.\n"
        "   — No uses conocimientos externos.\n"
        "   — No inventes datos no presentes.\n"
        "   — Usa también evidencias indirectas.\n"
        "\n"
        "2) PROCESO INTERNO (no mostrar):\n"
        "   — Identifica fragmentos relevantes.\n"
        "   — Combina toda la información aplicable.\n"
        "\n"
        "3) CRITERIO DE RESPUESTA:\n"
        "   — Responde si existe evidencia explícita o implícita.\n"
        "   — Solo responde 'No encontrado en Información' si no hay ninguna referencia relacionada.\n"
        "\n"
        "4) USO DE INFORMACIÓN:\n"
        "   — Utiliza el contenido, pero SIN mencionar pasos, numeraciones, fragmentos ni ubicaciones del texto.\n"
        "\n"
        "5) ESTILO:\n"
        "   — Respuesta en ESPAÑOL, directa y precisa.\n"
        "   — No incluyas enlaces salvo que provengan de INFORMACIÓN y empiecen por https, drive.google.com o docs.google.com.\n"
        "\n"
        "--------------------------------------------------\n"
        "1) INFORMACIÓN:\n"
        f"{articulo}\n"
        "\n"
        "--------------------------------------------------\n"
        "2) PREGUNTA ORIGINAL:\n"
        f"{question}\n"
        "2.1) PREGUNTA REFRASEADA:\n"
        f"{rephrased}\n"
        "\n"
        "--------------------------------------------------\n"
        "3) FORMATO DE SALIDA:\n"
        "\n"
        "<RESUMEN>\n"
        "Un único párrafo respondiendo la Pregunta.\n"
        "Si no existe evidencia suficiente: 'No encontrado en Información'.\n"
        "</RESUMEN>\n"
        "\n"
        "[ADICIONAL]\n"
        "- Lista de 2 a 15 ítems.\n"
        "- Incluye inferencias justificadas y aclaraciones útiles.\n"
        "- NO repitas el texto del RESUMEN.\n"
        "[/ADICIONAL]\n"
    )
    try:
        full_answer, timed_out = _invoke_chat_with_timeout(
            GPT_OSS_120_MODEL_ID,
            answer_prompt,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0
        )
        full_answer = (full_answer or "").strip()
        full_answer = remove_reasoning_blocks(full_answer)
        print("******************FULL ANSWER************************")
        print(full_answer)
        print("******************END FULL ANSWER************************")
    except Exception:
        full_answer = "(RESUMEN) No se pudo generar."

    mark(execution_times, 'generateAnswer_totalTime', t6)
    print(f"----------FIN - GENERAR RESPUESTA----------")
    print("***********************************************************")
    if not timed_out:
        t7 = tstart()
        print(f"----------INICIO - PARSEAR RESUMEN Y INFO ADICIONAL----------")
        resumen = ""
        info_adicional = ""

        # Normalización ligera de saltos HTML
        full_answer_norm = re.sub(r'(?i)<br\s*/?>', '\n', full_answer or "")

        # ======== RESUMEN ========
        m = re.search(
            r'''
            \[\s*RESUMEN\s*\]\s*
            (.*?)                                   # contenido
            \s*
            (?=
                \[
                /\s*RESUMEN\s*
                \]
                |
                \[\s*(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)\s*\]
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
                r'''
                <\s*RESUMEN\s*>\s*
                (.*?)                                # contenido
                \s*
                (?=
                    </\s*RESUMEN\s*>
                    |
                    </\s*>
                    |
                    \[\s*(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)\s*\]
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
                    if inner.upper() != "RESUMEN":
                        resumen = inner

        # ======== INFO_ADICIONAL ========
        m = re.search(
            r'''
            \[\s*(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)\s*\]\s*:?\s*
            (.*?)                                    # contenido
            \s*
            \[\s*/\s*(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)\s*\]
            ''',
            full_answer_norm, flags=re.IGNORECASE | re.DOTALL | re.UNICODE | re.VERBOSE
        )
        if m:
            info_adicional = m.group(1).strip()
        else:
            m = re.search(
                r'''
                \[\s*(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)\s*\]\s*:?\s*
                (.*?)                                 # contenido
                \s*
                (?=
                    </\s*adicional\s*>
                    | \[/\s*adicional\s*\]
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
                    if not re.fullmatch(r'(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)', candidate, flags=re.IGNORECASE):
                        info_adicional = candidate

        # ======== RECUPERACIÓN PARA CASO MALFORMADO: <RESUMEN> ... [/ADICIONAL] ... ========
        try:
            closing_add_re = r'\[/\s*(?:informaci[oó]n\s+adicional|adicional|detalles\s+adicionales)\s*\]'
            opening_res_re = r'<\s*RESUMEN\s*>'

            if not info_adicional:
                text = full_answer_norm or ""

                # Caso A: el RESUMEN quedó con un [/ADICIONAL] dentro (contaminado)
                if resumen and re.search(closing_add_re, resumen, flags=re.IGNORECASE):
                    parts = re.split(closing_add_re, resumen, maxsplit=1, flags=re.IGNORECASE)
                    resumen = (parts[0] or "").strip()
                    tail = (parts[1] if len(parts) > 1 else "").strip()

                    # Quitar un segundo cierre si aparece al final
                    tail = re.sub(closing_add_re + r'\s*$', "", tail, flags=re.IGNORECASE).strip()

                    if tail:
                        info_adicional = tail

                # Caso B: existe <RESUMEN> en el texto y uno o más cierres [/ADICIONAL] sin apertura
                if not info_adicional and re.search(opening_res_re, text, flags=re.IGNORECASE) and re.search(closing_add_re, text, flags=re.IGNORECASE):
                    after_resumen = re.split(opening_res_re, text, maxsplit=1, flags=re.IGNORECASE)[1]
                    before_close, *rest = re.split(closing_add_re, after_resumen, maxsplit=1, flags=re.IGNORECASE)
                    cand_resumen = (before_close or "").strip()
                    cand_tail = (rest[0] if rest else "").strip()

                    # Limpiar un posible segundo [/ADICIONAL] al final del tail
                    cand_tail = re.sub(closing_add_re + r'\s*$', "", cand_tail, flags=re.IGNORECASE).strip()

                    if cand_resumen and (not resumen or resumen.upper() == "RESUMEN"):
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

        # Limpieza Resumen
        resumen = remove_according_phrases(resumen)
        resumen = remove_resumen_tag(resumen)
        if resumen is not None and resumen != "":
            resumen = format_resumen(resumen)
            resumen = convertir_links_html(resumen)
        else:
            resumen = "Intermitencia de Servicio"
        print("*****************RESUMEN*************************")
        print(resumen)
        print("*****************FIN RESUMEN*************************")
        # Limpieza Info Adicional para Frontal
        print("*****************INFO ADICIONAL*************************")
        print(info_adicional)
        print("*****************FIN INFO ADICIONAL*************************")
        raw_additional = format_to_html(info_adicional)
        raw_additional = remove_according_phrases(raw_additional)

        # Limpieza Info Adicional para DynamoDB
        raw_additional_clean = clean_newlines(info_adicional)
        raw_additional_clean = remove_according_phrases(raw_additional_clean)

        mark(execution_times, 'parseAnswer', t7)
        print(f"----------FIN - PARSEAR RESUMEN Y INFO ADICIONAL----------")
        t8 = tstart()
        print("*************************************************************")
        print(f"----------INICIO - GUARDAR HISTORIAL----------")
        now_santiago = datetime.now(ZoneInfo("America/Santiago")).isoformat()
        total_time = sum(execution_times.values())
        question_item = {
            'id': str(uuid.uuid4()),
            'session_id': session_id or '',
            'question': question or '',
            'rephrased': rephrased or '',
            'category': category or 'Sin categoría',
            'answer': resumen or 'ERROR_GENERATION',
            'additional_information': raw_additional_clean or 'ERROR_GENERATION',
            'rag_files': rag_files or '',
            'date': now_santiago or '',
            'total_execution_time': str(total_time),
            'fragments': [],
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
        error_msg = "Error"
        if not resumen.strip(): 
            resumen = error_msg
        if not raw_additional.strip() or raw_additional.strip() == '[]':
            raw_additional = error_msg
            raw_additional_cleaned = error_msg

        response_body = {
            "sessionAttributes": {
                "client_transcript": question,
                "rephrased_message": rephrased,
                "kbid": KB_ID,
                "category": category or 'Sin categoría',
                "raw_message": resumen.replace("\n", "<br>"),
                "raw_additional": raw_additional or 'ERROR_GENERATION',
                "raw_additional_clean": raw_additional_clean.replace("\n", "") or 'Intermitencia de Servicio.',
                "fragments_used": texto_unico,
                "session_id": session_id,
                "rag_files": 'rag_files',
                "links": links,
                "metadata": metadata_list
            },
            "total_execution_time": total_time
        }
        if INCLUDE_EXEC_TIMES:
            response_body["execution_times"] = execution_times

        mark(execution_times, "buildResponse", t10)

    else:
        resumen = full_answer
        raw_additional = full_answer
        raw_additional_clean = full_answer
        category = category or 'Sin categoría'

        t10 = tstart()  # o time.time(), como estés midiendo
        total_time = time.time() - t0
        mark(execution_times, "buildResponse", t10)

        response_body = { 
            "sessionAttributes": {
                "client_transcript": question,
                "rephrased_message": rephrased,
                "kbid": KB_ID,
                "category": category,
                "raw_message": resumen.replace("\n", "<br>") if resumen else 'Intermitencia de Servicio.',
                "raw_additional": raw_additional or 'ERROR_GENERATION',
                "raw_additional_clean": raw_additional_clean.replace("\n", "") if raw_additional_clean else 'Intermitencia de Servicio.',
                "fragments_used": texto_unico,
                "session_id": session_id,
                "rag_files": 'rag_files',
                "links": links,
                "metadata": metadata_list
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