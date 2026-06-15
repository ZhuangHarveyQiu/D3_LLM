# -*- coding: utf-8 -*-
"""
d3_llm_essay_rating_colab.py
================================

Colab/Python script for collecting repeated LLM ratings of Dark Triad traits
from open-ended essay responses in stats_ready_long.csv.

Pipeline:
1. Load stats_ready_long.csv.
2. Create a hashed participant_id from run_id + recorded_at + ip + user_agent.
3. Keep only formal experiment participants/sessions with a valid 8-character confirmationCode.
4. Extract valid open-ended responses whose question_id starts with:
   machiv_clean, npi_clean, or srp_clean.
5. Keep only participants with exactly six valid essays:
   2 Machiavellianism, 2 Narcissism, 2 Psychopathy;
   1 broad and 1 specific prompt per trait.
6. Ask multiple LLMs, multiple repeated runs, to rate ALL THREE Dark Triad traits
   for each essay.
7. Save a long-format file: one row per essay x model x run.

Important privacy note:
- ip and user_agent are used only to construct hashed participant IDs.
- They are NOT sent to LLM APIs.
- They are dropped from saved LLM input/output files after hashing.
"""

# ====== Optional dependency installation ======
# This works in Colab and regular Python. Comment out if your environment already has these.
import sys, subprocess, importlib.util

def _install_if_missing(package_name: str, pip_name: str = None):
    if importlib.util.find_spec(package_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or package_name])

_install_if_missing("pandas")
_install_if_missing("huggingface_hub")
_install_if_missing("openai")

import os, re, json, time, random, traceback, hashlib, glob, shutil
from typing import Dict, Optional, List, Tuple, Any
import pandas as pd
from huggingface_hub import InferenceClient, HfApi
from openai import OpenAI as _OpenAI

# ====== Paths ======
IN_PATH = "/content/stats_ready_long.csv"
ESSAY_INPUT_PATH = "/content/d3_essay_rating_input_FORMAL.csv"
OUT_PATH = "/content/d3_llm_essay_ratings_long_FORMAL.csv"
EXCLUDED_PATH = "/content/d3_excluded_participants_FORMAL.csv"

# ====== Config ======
SCORE_MIN, SCORE_MAX = 1, 5
CONF_MIN, CONF_MAX = 1, 5
N_RUNS_PER_MODEL = 4
LLM_TEMPERATURE = 0.5  # Deliberate design choice for repeated-rating reliability; document before production.
MAX_ESSAYS = None      # For pilot testing, set e.g. 20. For full run, keep None.
SLEEP_BETWEEN_CALLS_SEC = 0.0
SAVE_EVERY_N_ROWS = 25
PROMPT_VERSION = "d3_response_only_v2_2026-06-11"
SCRIPT_VERSION = "d3_llm_essay_rating_colab_revised_v2"
VALID_CONFIRMATION_RE = r"^[A-Z0-9]{8}$"

# Output mode. Pilot runs are structurally isolated from formal outputs.
# For a pilot, set MAX_ESSAYS to a small integer such as 10 or 20.
RUN_MODE = "PILOT" if MAX_ESSAYS is not None else "FORMAL"

if RUN_MODE == "PILOT":
    OUT_PATH = OUT_PATH.replace("_FORMAL.csv", "_PILOT.csv")
    ESSAY_INPUT_PATH = ESSAY_INPUT_PATH.replace("_FORMAL.csv", "_PILOT.csv")
    EXCLUDED_PATH = EXCLUDED_PATH.replace("_FORMAL.csv", "_PILOT.csv")

print("RUN_MODE:", RUN_MODE)
print("OUT_PATH:", OUT_PATH)
print("ESSAY_INPUT_PATH:", ESSAY_INPUT_PATH)
print("EXCLUDED_PATH:", EXCLUDED_PATH)

# Failed/unparsed rows are never treated as completed for resume purposes.

MODEL_NAMES = [
    "LLaMA-3",
    "Qwen",
    # "Mistral-7B",
    # "Mixtral-8x7B",
    "GPT-OSS-120B",
    "GPT-5",
    "Grok-4",
]

# ====== Save to Google Drive ======
try:
    from google.colab import drive, userdata
    drive.mount('/content/drive')
    drive_root_options = glob.glob('/content/drive/My*')
    drive_root = drive_root_options[0] if drive_root_options else "/content/drive/MyDrive"
    BASE_OUTPUT_DIR = os.path.join(drive_root, "MyDrive2026_D3_LLM_ratings")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Drive output folder: {BASE_OUTPUT_DIR}")

    HF_TOKEN = userdata.get('HF_TOKEN')
    OPENAI_API_KEY = userdata.get('openai')
    XAI_API_KEY = userdata.get('Xai')
except Exception as e:
    print("Google Colab Drive/userdata not available; using environment variables only.", repr(e))
    BASE_OUTPUT_DIR = None
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_MAIN")
    XAI_API_KEY = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")

if HF_TOKEN and os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

print("HF token present:", bool(HF_TOKEN or os.getenv("HUGGINGFACEHUB_API_TOKEN")))
print("OpenAI key present:", bool(OPENAI_API_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_MAIN")))
print("xAI key present:", bool(XAI_API_KEY or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")))

# ====== Retry Helper ======
def try_retry(fn, retries=2, backoff=1.6, jitter=0.25, retry_if=lambda e: True):
    last_err = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i == retries or not retry_if(e):
                print("no-retry or max retries hit:", repr(e))
                traceback.print_exc()
                break
            delay = max(0.0, (backoff ** i) + random.uniform(-jitter, jitter))
            print(f"retry {i + 1}/{retries} in {delay:.2f}s after error: {repr(e)}")
            time.sleep(delay)
    raise last_err

# ====== Model IDs / Endpoints ======
HF_MODEL_LLAMA3 = os.getenv("HF_MODEL_LLAMA3", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_ENDPOINT_URL_LLAMA3 = os.getenv("HF_ENDPOINT_URL_LLAMA3", "").strip()

HF_MODEL_QWEN = os.getenv("HF_MODEL_QWEN", "Qwen/Qwen3.5-397B-A17B")
HF_ENDPOINT_URL_QWEN = os.getenv("HF_ENDPOINT_URL_QWEN", "").strip()

HF_MODEL_MISTRAL7B = os.getenv("HF_MODEL_MISTRAL7B", "mistralai/Mistral-7B-Instruct-v0.2")
HF_ENDPOINT_URL_MISTRAL7B = os.getenv("HF_ENDPOINT_URL_MISTRAL7B", "").strip()

HF_MODEL_GPTOSS120B = os.getenv("HF_MODEL_GPTOSS120B", "openai/gpt-oss-120b")
HF_API = HfApi()

# ====== D3 Rating Prompt ======
D3_RATING_PROMPT = """You are an expert psychological text rater.

You will read ONE participant response from an open-ended personality study.
Your task is to rate the extent to which the response provides textual evidence for three Dark Triad traits.

Important rules:
- Rate ONLY the participant's response. Do not infer traits from the study design, prompt label, or what you think the original scenario might have been.
- Do not diagnose the participant. These are text-based ratings of expressed reasoning and self-presentation, not clinical judgments.
- Use the full 1-5 scale when appropriate.
- If there is no clear evidence for a trait, give a low score.
- If the evidence is mixed or ambiguous, use the middle of the scale.
- Return ONLY valid JSON and no extra text.

Trait definitions:
1. Machiavellianism: strategic manipulation, calculated self-interest, distrust, instrumental use of others, willingness to conceal motives or exploit social situations for advantage.
2. Narcissism: grandiosity, entitlement, superiority, desire for admiration, need for recognition, belief in specialness or authority over others.
3. Psychopathy: callousness, low empathy/remorse, impulsive risk-taking, thrill seeking, aggression, disregard for rules or others' welfare.

Rating scale for each trait:
1 = no evidence or clearly low evidence
2 = weak evidence
3 = mixed, ambiguous, or moderate evidence
4 = strong evidence
5 = very strong evidence

Confidence scale:
1 = very low confidence
2 = low confidence
3 = moderate confidence
4 = high confidence
5 = very high confidence

Return exactly this JSON shape:
{{
  "ratings": {{
    "machiavellianism": <1-5>,
    "narcissism": <1-5>,
    "psychopathy": <1-5>
  }},
  "confidence": <1-5>,
  "evidence": "one brief sentence explaining the rating based only on the response"
}}

PARTICIPANT RESPONSE:
{essay_text}
"""

# ====== Parsing helpers ======
def _clip_int(v, lo=SCORE_MIN, hi=SCORE_MAX):
    try:
        return int(min(max(int(round(float(v))), lo), hi))
    except Exception:
        return None

def _clip_conf(v):
    try:
        return int(min(max(int(round(float(v))), CONF_MIN), CONF_MAX))
    except Exception:
        return None

def _strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)

def _find_balanced_json_candidates(s: str) -> List[str]:
    """Find possible balanced JSON objects in a string."""
    s = _strip_code_fences(s)
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    candidates = []
    for start in starts:
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(s[start:j+1])
                        break
    return candidates

def _clean_json_like(block: str) -> str:
    b = block.strip().replace("\r\n", "\n")
    try:
        json.loads(b)
        return b
    except Exception:
        pass
    b = re.sub(r"(?<!\\)'", '"', b)
    b = re.sub(r'(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', b)
    b = re.sub(r",\s*(\}|])", r"\1", b)
    return b

def _try_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    candidates = _find_balanced_json_candidates(raw)
    # Prefer blocks containing ratings or trait names.
    candidates = sorted(
        candidates,
        key=lambda x: int(any(k in x.lower() for k in ["ratings", "machiavellianism", "narcissism", "psychopathy"])),
        reverse=True,
    )
    for cand in candidates:
        try:
            obj = json.loads(_clean_json_like(cand))
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def parse_d3_ratings(raw_text: str) -> Dict[str, Any]:
    raw = raw_text or ""
    obj = _try_parse_json(raw)
    ratings = {}
    confidence = None
    evidence = ""
    parse_method = "json"

    if obj:
        ratings_obj = obj.get("ratings", obj)
        if isinstance(ratings_obj, dict):
            ratings["machiavellianism"] = _clip_int(ratings_obj.get("machiavellianism"))
            ratings["narcissism"] = _clip_int(ratings_obj.get("narcissism"))
            ratings["psychopathy"] = _clip_int(ratings_obj.get("psychopathy"))
        confidence = _clip_conf(obj.get("confidence"))
        evidence = str(obj.get("evidence", ""))[:500]
    else:
        parse_method = "regex"
        patterns = {
            "machiavellianism": r'machiavellianism[^0-9]{0,30}([1-5])',
            "narcissism": r'narcissism[^0-9]{0,30}([1-5])',
            "psychopathy": r'psychopathy[^0-9]{0,30}([1-5])',
        }
        for k, p in patterns.items():
            m = re.search(p, raw, flags=re.IGNORECASE)
            ratings[k] = _clip_int(m.group(1)) if m else None
        mconf = re.search(r'confidence[^0-9]{0,30}([1-5])', raw, flags=re.IGNORECASE)
        confidence = _clip_conf(mconf.group(1)) if mconf else None

    parse_success = all(ratings.get(k) is not None for k in ["machiavellianism", "narcissism", "psychopathy"])
    return {
        "machiavellianism": ratings.get("machiavellianism"),
        "narcissism": ratings.get("narcissism"),
        "psychopathy": ratings.get("psychopathy"),
        "confidence": confidence,
        "evidence": evidence,
        "parse_success": bool(parse_success),
        "parse_method": parse_method,
    }

# Keep the original name used by older model-call functions.
def parse_scores(raw_text: str) -> Dict[str, Any]:
    return parse_d3_ratings(raw_text)

# ====== HF provider helpers ======
def get_provider_map(model_id: str) -> Any:
    try:
        info = HF_API.model_info(model_id, expand=["inferenceProviderMapping"])
        mapping = getattr(info, "inference_provider_mapping", None)
        return mapping or {}
    except Exception:
        return {}

def get_supported_tasks(mapping) -> Dict[str, List[str]]:
    if isinstance(mapping, dict):
        return mapping
    elif isinstance(mapping, list):
        provider_map = {}
        for mp in mapping:
            provider = getattr(mp, 'provider', '')
            task = getattr(mp, 'task', None)
            if provider and task:
                provider_map.setdefault(provider, []).append(task)
        return provider_map
    return {}

def pick_hosted_provider_for(model_id: str, preferred_tasks: List[str]) -> Optional[Tuple[str, str]]:
    provider_map = get_supported_tasks(get_provider_map(model_id))
    for provider, tasks in provider_map.items():
        for task in preferred_tasks:
            if task in tasks:
                return (provider, task)
    return None

# ====== OpenAI-style text extraction ======
def _extract_openai_chat_text(resp):
    try:
        ch = resp.choices[0]
    except Exception:
        return str(resp or "") or ""
    msg = getattr(ch, "message", None)
    if msg is None and isinstance(ch, dict):
        msg = ch.get("message", {})
    try:
        content = getattr(msg, "content", None) if msg is not None else None
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, (list, tuple)) and content:
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(p.get("text") or p.get("content") or "")
                else:
                    parts.append(getattr(p, "text", "") or getattr(p, "content", ""))
            joined = " ".join([pt for pt in parts if pt])
            if joined.strip():
                return joined
    except Exception:
        pass
    try:
        if hasattr(ch, "text") and isinstance(ch.text, str) and ch.text.strip():
            return ch.text
        if isinstance(ch, dict) and isinstance(ch.get("text"), str) and ch["text"].strip():
            return ch["text"]
    except Exception:
        pass
    # Deliberately do NOT fall back to any reasoning field.
    # Reasoning text can contain stray numbers and must not be parsed as ratings.
    return ""

# ====== LLaMA-3 ======
def _client_llama3() -> InferenceClient:
    token = HF_TOKEN or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if HF_ENDPOINT_URL_LLAMA3:
        return InferenceClient(model=None, token=token, timeout=120, base_url=HF_ENDPOINT_URL_LLAMA3.rstrip("/"))
    return InferenceClient(model=HF_MODEL_LLAMA3, token=token, timeout=120)

def call_llama3(prompt: str):
    try:
        client = _client_llama3()
        def _chat():
            return client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512, temperature=LLM_TEMPERATURE,
            )
        try:
            resp = try_retry(_chat, retries=2)
            raw = resp.choices[0].message["content"] if hasattr(resp, "choices") else str(resp)
        except Exception:
            def _text():
                return client.text_generation(prompt, max_new_tokens=512, do_sample=True, temperature=LLM_TEMPERATURE, return_full_text=False)
            resp = try_retry(_text, retries=2)
            raw = str(resp)
        print("LLaMA-3 RAW:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "llama3"
    except Exception as e:
        print("LLaMA-3 ERROR:", repr(e))
        return _empty_rating(), f"error: {repr(e)}", "error"

# ====== Qwen ======
def _openai_qwen_client():
    token = (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or HF_TOKEN)
    if not token:
        raise RuntimeError("Missing HF token. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.")
    base_url = HF_ENDPOINT_URL_QWEN.rstrip("/") if HF_ENDPOINT_URL_QWEN else "https://router.huggingface.co/v1"
    return _OpenAI(base_url=base_url, api_key=token, timeout=120.0)

def call_qwen(prompt: str):
    system_msg = (
        "You are a strict JSON generator. Output ONLY valid JSON in exactly this shape: "
        '{"ratings":{"machiavellianism":INT,"narcissism":INT,"psychopathy":INT},"confidence":INT,"evidence":"TEXT"}. '
        f"INTs must be in [{SCORE_MIN},{SCORE_MAX}] for ratings and [{CONF_MIN},{CONF_MAX}] for confidence."
    )
    try:
        client = _openai_qwen_client()
        def _chat_router():
            return client.chat.completions.create(
                model=HF_MODEL_QWEN,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=512,
                extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            )
        resp = try_retry(_chat_router, retries=2)
        raw = _extract_openai_chat_text(resp) or ""
        print("Qwen RAW:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "qwen-router"
    except Exception as e:
        print("Qwen router path failed; trying HF fallback:", repr(e))
    try:
        token = HF_TOKEN or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        client = InferenceClient(model=HF_MODEL_QWEN, token=token, timeout=120)
        def _chat_hf():
            return client.chat_completion(
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                max_tokens=512, temperature=LLM_TEMPERATURE,
            )
        resp = try_retry(_chat_hf, retries=2)
        try:
            raw = resp.choices[0].message["content"]
        except Exception:
            raw = str(resp)
        print("Qwen RAW HF fallback:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "qwen-hf-fallback"
    except Exception as e:
        print("Qwen ERROR:", repr(e))
        return _empty_rating(), f"error: {repr(e)}", "error"

# ====== Mistral-7B, optional ======
def call_mistral7b(prompt: str):
    try:
        token = HF_TOKEN or os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
        client = InferenceClient(model=HF_MODEL_MISTRAL7B, token=token, timeout=120)
        def _chat():
            return client.chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=512, temperature=LLM_TEMPERATURE)
        resp = try_retry(_chat, retries=2)
        raw = resp.choices[0].message["content"] if hasattr(resp, "choices") else str(resp)
        print("Mistral-7B RAW:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "mistral7b"
    except Exception as e:
        print("Mistral-7B ERROR:", repr(e))
        return _empty_rating(), f"error: {repr(e)}", "error"

# ====== GPT-OSS-120B via HF router ======
def _openai_hf_router_client():
    token = (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or HF_TOKEN)
    if not token:
        raise RuntimeError("Missing HF token. Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.")
    return _OpenAI(base_url="https://router.huggingface.co/v1", api_key=token, timeout=60.0)

def call_gptoss120b(prompt: str):
    try:
        client = _openai_hf_router_client()
    except Exception as e:
        raw = f"client-init-error: {repr(e)}"
        print("GPT-OSS-120B INIT ERROR:", raw)
        return _empty_rating(), raw, "client-init-error"
    system_msg = (
        "You are a strict JSON generator. Output ONLY valid JSON in exactly this shape: "
        '{"ratings":{"machiavellianism":INT,"narcissism":INT,"psychopathy":INT},"confidence":INT,"evidence":"TEXT"}. '
        "No chain-of-thought or extra text."
    )
    def _chat():
        return client.chat.completions.create(
            model=HF_MODEL_GPTOSS120B,
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=512,
        )
    try:
        resp = try_retry(_chat, retries=3, backoff=2.0, jitter=0.5)
        raw = _extract_openai_chat_text(resp) or ""
        print("GPT-OSS-120B RAW:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "openai-hf-router"
    except Exception as e:
        tb = traceback.format_exc()
        raw = f"exception: {repr(e)}\n{tb}"
        print("GPT-OSS-120B ERROR:", repr(e))
        return _empty_rating(), raw, "error"

# ====== OpenAI GPT-5 direct ======
def _openai_gpt_client():
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_MAIN") or OPENAI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("Missing OpenAI API key. Set OPENAI_API_KEY or OPENAI_API_KEY_MAIN.")
    return _OpenAI(api_key=api_key, timeout=60.0)

def call_gpt5(prompt: str):
    try:
        client = _openai_gpt_client()
        def _call():
            return client.responses.create(
                model="gpt-5.2",
                input=[
                    {"role": "developer", "content": "Return ONLY valid JSON with Dark Triad ratings in the exact format requested."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=512,
                temperature=LLM_TEMPERATURE,
                top_p=1.0,
                store=False,
            )
        resp = try_retry(_call, retries=1)

        # Critical safety choice:
        # If GPT-5 produces no visible final answer, do NOT fall back to str(resp).
        # str(resp) may contain metadata/reasoning/object fields with stray numbers,
        # which could be accidentally parsed as ratings. Empty raw text causes a
        # clean parse failure and will be retried by the resume logic.
        raw = getattr(resp, "output_text", "") or ""

        print("GPT-5 RAW:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "openai-responses"
    except Exception as e:
        print("GPT-5 ERROR:", repr(e))
        return _empty_rating(), f"error: {repr(e)}", "error"

# ====== xAI Grok-4 ======
def _xai_grok_client():
    api_key = (os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or XAI_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("Missing xAI key. Set XAI_API_KEY or GROK_API_KEY.")
    return _OpenAI(base_url="https://api.x.ai/v1", api_key=api_key, timeout=60.0)

def call_grok4(prompt: str):
    try:
        client = _xai_grok_client()
        def _chat():
            return client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                top_p=1.0,
                max_tokens=512,
            )
        resp = try_retry(_chat, retries=1)
        raw = _extract_openai_chat_text(resp) or ""
        print("Grok-4 RAW:", raw[:250].replace("\n", " ") + ("..." if len(raw) > 250 else ""))
        return parse_scores(raw), raw, "xai-chat"
    except Exception as e:
        print("Grok-4 ERROR:", repr(e))
        return _empty_rating(), f"error: {repr(e)}", "error"

def _empty_rating():
    return {
        "machiavellianism": None,
        "narcissism": None,
        "psychopathy": None,
        "confidence": None,
        "evidence": "",
        "parse_success": False,
        "parse_method": "error",
    }

def call_model(model_name: str, prompt: str):
    if model_name == "LLaMA-3":
        return call_llama3(prompt)
    elif model_name == "Qwen":
        return call_qwen(prompt)
    elif model_name == "Mistral-7B":
        return call_mistral7b(prompt)
    elif model_name == "GPT-OSS-120B":
        return call_gptoss120b(prompt)
    elif model_name == "GPT-5":
        return call_gpt5(prompt)
    elif model_name == "Grok-4":
        return call_grok4(prompt)
    else:
        print(f"Unknown model_name: {model_name}")
        return _empty_rating(), "", "unknown-model"

# ====== Data preparation ======
def _valid_confirmation_code(x) -> bool:
    """Return True only for real completion codes: 8 uppercase alphanumeric characters."""
    if pd.isna(x):
        return False
    s = str(x).strip()
    return bool(re.fullmatch(VALID_CONFIRMATION_RE, s))

def _nonempty_code(x) -> bool:
    """Backward-compatible alias. Use strict validation for formal analyses."""
    return _valid_confirmation_code(x)

def make_participant_id(row) -> str:
    raw = "|".join([
        str(row.get("run_id", "")),
        str(row.get("recorded_at", "")),
        str(row.get("ip", "")),
        str(row.get("user_agent", "")),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

def infer_target_trait(question_id: str) -> Optional[str]:
    q = str(question_id)
    if q.startswith("machiv_clean"):
        return "Machiavellianism"
    if q.startswith("npi_clean"):
        return "Narcissism"
    if q.startswith("srp_clean"):
        return "Psychopathy"
    return None

def infer_prompt_type(question_id: str) -> Optional[str]:
    q = str(question_id).lower()
    if q.endswith("_broad"):
        return "broad"
    if q.endswith("_specific"):
        return "specific"
    return None

def word_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"\b\S+\b", text.strip()))

def prepare_essay_input(in_path: str = IN_PATH, out_path: str = ESSAY_INPUT_PATH) -> pd.DataFrame:
    """Create the formal essay-rating input file.

    Inclusion rules:
    1. participant/session has at least one valid confirmationCode matching ^[A-Z0-9]{8}$;
    2. essay rows are machiv_clean/npi_clean/srp_clean;
    3. explicitly invalid rows (valid == False) are removed;
    4. participant has exactly six valid essays: 2 per D3 trait, with one broad and one specific prompt per trait.

    Privacy rule:
    ip and user_agent are used to hash participant_id and then dropped from all saved outputs.
    """
    df = pd.read_csv(in_path)
    required = {
        "run_id", "confirmationCode", "question_id", "value", "valid", "trial_index",
        "trial_type", "ip", "recorded_at", "user_agent", "total_rt"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    df["participant_id"] = df.apply(make_participant_id, axis=1)
    df["valid_confirmation_code"] = df["confirmationCode"].apply(_valid_confirmation_code)
    df["confirmationCode_clean"] = df["confirmationCode"].where(df["valid_confirmation_code"], pd.NA)

    # Formal participants: sessions with at least one real 8-character completion code.
    confirmed_ids = df.loc[df["valid_confirmation_code"], "participant_id"].dropna().unique()
    df_confirmed = df[df["participant_id"].isin(confirmed_ids)].copy()

    # Per-session metadata for audit and later merging. Do not retain ip/user_agent in saved files.
    session_meta = (
        df.groupby("participant_id", as_index=False)
          .agg(
              run_id=("run_id", "first"),
              recorded_at=("recorded_at", "first"),
              confirmationCode=("confirmationCode_clean", lambda x: x.dropna().iloc[0] if len(x.dropna()) else pd.NA),
              any_valid_confirmation_code=("valid_confirmation_code", "max"),
              raw_confirmation_values=("confirmationCode", lambda x: " | ".join(sorted(set(str(v).strip() for v in x.dropna() if str(v).strip()))[:5])),
          )
    )

    # Essay rows.
    essay_mask = df_confirmed["question_id"].astype(str).str.startswith(("machiv_clean", "npi_clean", "srp_clean"))
    essays = df_confirmed[essay_mask].copy()

    # Remove explicitly invalid essays. Null valid is not treated as invalid, but essay rows should usually be True.
    invalid_mask = essays["valid"].astype(str).str.lower().eq("false")
    essays = essays[~invalid_mask].copy()

    essays = essays[essays["value"].notna() & (essays["value"].astype(str).str.strip() != "")].copy()
    essays["target_trait"] = essays["question_id"].apply(infer_target_trait)
    essays["prompt_type"] = essays["question_id"].apply(infer_prompt_type)
    essays["essay_text"] = essays["value"].astype(str)
    essays["word_count"] = essays["essay_text"].apply(word_count)

    # If duplicates remain for same participant/question, keep latest trial_index.
    essays["trial_index_num"] = pd.to_numeric(essays["trial_index"], errors="coerce")
    essays = essays.sort_values(["participant_id", "question_id", "trial_index_num"])
    essays = essays.drop_duplicates(["participant_id", "question_id"], keep="last")

    def is_complete(group: pd.DataFrame) -> bool:
        if len(group) != 6:
            return False
        trait_counts = group["target_trait"].value_counts().to_dict()
        if trait_counts.get("Machiavellianism", 0) != 2:
            return False
        if trait_counts.get("Narcissism", 0) != 2:
            return False
        if trait_counts.get("Psychopathy", 0) != 2:
            return False
        for trait in ["Machiavellianism", "Narcissism", "Psychopathy"]:
            sub = group[group["target_trait"] == trait]
            pt = sub["prompt_type"].value_counts().to_dict()
            if pt.get("broad", 0) != 1 or pt.get("specific", 0) != 1:
                return False
        return True

    if essays.empty:
        completeness = pd.DataFrame(columns=["participant_id", "complete_six_essays"])
    else:
        completeness = essays.groupby("participant_id").apply(is_complete).reset_index(name="complete_six_essays")
    keep_ids = completeness.loc[completeness["complete_six_essays"] == True, "participant_id"].tolist()

    # Audit all sessions, not only confirmed ones, so malformed codes are visible.
    n_essays = essays.groupby("participant_id").size().reset_index(name="n_valid_essays") if not essays.empty else pd.DataFrame(columns=["participant_id", "n_valid_essays"])
    audit = session_meta.merge(n_essays, on="participant_id", how="left")
    audit["n_valid_essays"] = audit["n_valid_essays"].fillna(0).astype(int)
    audit = audit.merge(completeness, on="participant_id", how="left")
    audit["complete_six_essays"] = audit["complete_six_essays"].fillna(False)
    audit["included_formal_experiment"] = audit["participant_id"].isin(keep_ids)
    audit["exclusion_reason"] = ""
    audit.loc[~audit["any_valid_confirmation_code"].astype(bool), "exclusion_reason"] = "no_valid_8char_confirmation_code"
    audit.loc[audit["any_valid_confirmation_code"].astype(bool) & ~audit["complete_six_essays"].astype(bool), "exclusion_reason"] = "valid_code_but_not_exactly_6_balanced_valid_essays"
    audit.loc[audit["included_formal_experiment"], "exclusion_reason"] = "included"
    audit.to_csv(EXCLUDED_PATH, index=False)

    essays = essays[essays["participant_id"].isin(keep_ids)].copy()
    essays = essays.sort_values(["participant_id", "target_trait", "prompt_type", "question_id"]).reset_index(drop=True)
    essays["essay_number"] = essays.groupby("participant_id").cumcount() + 1
    essays["essay_id"] = essays["participant_id"] + "_essay" + essays["essay_number"].astype(str).str.zfill(2)
    essays["prompt_version"] = PROMPT_VERSION
    essays["script_version"] = SCRIPT_VERSION

    # Save only privacy-safe columns. ip and user_agent have already done their job in participant_id hashing.
    cols = [
        "essay_id", "participant_id", "run_id", "confirmationCode", "question_id",
        "target_trait", "prompt_type", "essay_text", "word_count", "total_rt",
        "valid", "trial_index", "trial_type", "recorded_at", "prompt_version", "script_version",
    ]
    essays[cols].to_csv(out_path, index=False)

    print("Sessions with valid 8-character confirmation code:", len(confirmed_ids))
    print("Formal complete participant sessions kept:", len(keep_ids))
    print("Essays kept:", len(essays))
    print("Wrote privacy-safe essay input:", out_path)
    print("Wrote exclusion audit:", EXCLUDED_PATH)
    return essays[cols]

# ====== Run LLM ratings ======
def load_existing_results(out_path: str) -> pd.DataFrame:
    if os.path.exists(out_path):
        try:
            return pd.read_csv(out_path)
        except Exception:
            pass
    return pd.DataFrame()

def already_done_keys(existing: pd.DataFrame) -> set:
    """Resume only successful rows. Failed/unparsed rows are retried on the next run."""
    if existing.empty:
        return set()
    needed = {"essay_id", "model", "run", "parse_success"}
    if not needed.issubset(existing.columns):
        return set()
    ok = existing["parse_success"].astype(str).str.lower().isin({"true", "1"})
    for col in ["llm_machiavellianism", "llm_narcissism", "llm_psychopathy"]:
        if col in existing.columns:
            ok &= existing[col].notna()
    good = existing.loc[ok].copy()
    if good.empty:
        return set()
    return set(zip(good["essay_id"].astype(str), good["model"].astype(str), good["run"].astype(int)))

def save_results_incremental(rows: List[Dict[str, Any]], out_path: str):
    """Append buffered rows and de-duplicate, preferring the latest row.

    This is still CSV-based for Colab convenience, but writes are buffered by
    SAVE_EVERY_N_ROWS to avoid re-reading/re-writing after every API call.
    """
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        out = pd.concat([old, new_df], ignore_index=True)
        out = out.drop_duplicates(["essay_id", "model", "run"], keep="last")
    else:
        out = new_df
    out.to_csv(out_path, index=False)


def actual_model_id(model_name: str) -> str:
    return {
        "LLaMA-3": HF_MODEL_LLAMA3,
        "Qwen": HF_MODEL_QWEN,
        "Mistral-7B": HF_MODEL_MISTRAL7B,
        "GPT-OSS-120B": HF_MODEL_GPTOSS120B,
        "GPT-5": "gpt-5.2",
        "Grok-4": "grok-4",
    }.get(model_name, model_name)


def run_llm_ratings(essay_df: pd.DataFrame, out_path: str = OUT_PATH):
    if MAX_ESSAYS is not None:
        essay_df = essay_df.head(MAX_ESSAYS).copy()
        print(f"Pilot mode: rating first {len(essay_df)} essays only.")

    existing = load_existing_results(out_path)
    done = already_done_keys(existing)
    buffer_rows = []

    total_jobs = len(essay_df) * len(MODEL_NAMES) * N_RUNS_PER_MODEL
    job_i = 0
    for essay_idx, (_, row) in enumerate(essay_df.iterrows(), start=1):
        essay_text = str(row["essay_text"])
        prompt = D3_RATING_PROMPT.format(essay_text=essay_text.strip())

        for model in MODEL_NAMES:
            for run in range(1, N_RUNS_PER_MODEL + 1):
                job_i += 1
                key = (str(row["essay_id"]), str(model), int(run))
                if key in done:
                    print(f"Skipping existing {key}")
                    continue

                print(f"\n===== Job {job_i}/{total_jobs}: essay {essay_idx}/{len(essay_df)}, essay_id={row['essay_id']}, model={model}, run={run} =====")
                scores, raw, parse_method_call = call_model(model, prompt)
                if not isinstance(scores, dict):
                    scores = _empty_rating()

                out_row = {
                    "essay_id": row["essay_id"],
                    "participant_id": row["participant_id"],
                    "question_id": row["question_id"],
                    "target_trait": row["target_trait"],
                    "prompt_type": row["prompt_type"],
                    "word_count": row["word_count"],
                    "total_rt": row["total_rt"],
                    "model": model,
                    "model_id_actual": actual_model_id(model),
                    "run": run,
                    "temperature": LLM_TEMPERATURE,
                    "top_p": 1.0,
                    "prompt_version": PROMPT_VERSION,
                    "script_version": SCRIPT_VERSION,
                    "rating_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                    "llm_machiavellianism": scores.get("machiavellianism"),
                    "llm_narcissism": scores.get("narcissism"),
                    "llm_psychopathy": scores.get("psychopathy"),
                    "llm_confidence": scores.get("confidence"),
                    "llm_evidence": scores.get("evidence", ""),
                    "parse_success": scores.get("parse_success", False),
                    "parse_method": scores.get("parse_method", parse_method_call),
                    "call_method": parse_method_call,
                    "raw_response": raw,
                }
                buffer_rows.append(out_row)

                # Save buffered rows to protect against crashes/disconnects without O(n^2) writes.
                if bool(out_row.get("parse_success")):
                    done.add(key)

                if len(buffer_rows) >= SAVE_EVERY_N_ROWS:
                    save_results_incremental(buffer_rows, out_path)
                    buffer_rows = []

                if SLEEP_BETWEEN_CALLS_SEC > 0:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    # Flush remaining buffered rows.
    save_results_incremental(buffer_rows, out_path)

    print("\nFinished LLM ratings. Output:", out_path)
    if BASE_OUTPUT_DIR and os.path.exists(out_path):
        dest = os.path.join(BASE_OUTPUT_DIR, os.path.basename(out_path))
        shutil.copy(out_path, dest)
        print("Copied ratings file to Drive:", dest)
    if BASE_OUTPUT_DIR and os.path.exists(ESSAY_INPUT_PATH):
        dest = os.path.join(BASE_OUTPUT_DIR, os.path.basename(ESSAY_INPUT_PATH))
        shutil.copy(ESSAY_INPUT_PATH, dest)
        print("Copied essay input file to Drive:", dest)
    if BASE_OUTPUT_DIR and os.path.exists(EXCLUDED_PATH):
        dest = os.path.join(BASE_OUTPUT_DIR, os.path.basename(EXCLUDED_PATH))
        shutil.copy(EXCLUDED_PATH, dest)
        print("Copied exclusion audit file to Drive:", dest)

# ====== Main ======
if __name__ == "__main__":
    essay_input = prepare_essay_input(IN_PATH, ESSAY_INPUT_PATH)
    print("Essay input preview:")
    print(essay_input.head())
    run_llm_ratings(essay_input, OUT_PATH)
