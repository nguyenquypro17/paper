"""
vlm_common.py -- shared helpers for render_concepts.py / vlm_name.py / vlm_predict.py /
label_factor.py.

Needs transformers >= 4.52 (5.x fine) and torchvision; 4-bit needs bitsandbytes.
"""

import json
import os
import re
import time

import torch
from PIL import Image

MODELS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
    "gemma": "google/gemma-3-12b-it",
}

# Neutral descriptions of the environment and of the ONE view shown to the VLM
# (MiniGrid: the agent's ego-centric view, the only input of the concept;
#  CartPole / Atari: the full rendered frame). No rules, no actions, no hints about
# what concepts should mean.
_MG_VIEW = ("Each image is the agent's 7x7 ego-centric view: the agent is the red triangle "
            "in the bottom-centre cell and always faces the top of the image; cells it "
            "cannot see are black; an object the agent is carrying is drawn in the agent's own cell.")
ENV_CONTEXT = {
    "MiniGrid-DoorKey-6x6-v0":
        "A small grid world containing a yellow key, a yellow door in a wall, and a green goal square. " + _MG_VIEW,
    "MiniGrid-Dynamic-Obstacles-5x5-v0":
        "A small grid world containing moving blue balls and a green goal square. " + _MG_VIEW,
    "PixelCartPole":
        "CartPole: a pole is hinged on a cart that moves left or right along a track. "
        "Each image is the rendered frame of the scene.",
    "PongNoFrameskip-v4":
        "Atari Pong: the agent controls the right paddle, the opponent the left paddle, and a small "
        "ball moves between them. Each image is the rendered game screen.",
    "BoxingNoFrameskip-v4":
        "Atari Boxing, top-down view of a ring: the agent controls the white boxer, the opponent is "
        "black. Each image is the rendered game screen.",
}

# Positive-control statements: factor -> (statement, binarizer). Only factors that are
# visible in the single view shown to the VLM (no velocities, no agent_dir).
FACTOR_TESTS = {
    "DoorKey": {
        "has_key": ("The agent is carrying the key.", lambda v: v > 0.5),
        "door_open": ("The door is open.", lambda v: v > 0.5),
        "key_in_front": ("The key is in the cell directly in front of the agent.", lambda v: v > 0.5),
        "door_in_front": ("The door is in the cell directly in front of the agent.", lambda v: v > 0.5),
        "wall_ahead": ("A wall is in the cell directly in front of the agent.", lambda v: v > 0.5),
        # goal_visible dropped: identical to door_open in DoorKey-6x6 (goal is behind the door)
    },
    "Dynamic-Obstacles": {
        "obstacle_ahead": ("A blue ball is in the cell directly in front of the agent.", lambda v: v > 0.5),
        "obstacle_left": ("A blue ball is in the cell directly to the left of the agent.", lambda v: v > 0.5),
        "obstacle_right": ("A blue ball is in the cell directly to the right of the agent.", lambda v: v > 0.5),
    },
    "PixelCartPole": {
        "pole_angle": ("The pole is leaning to the right.", lambda v: v > 0),
        "cart_pos": ("The cart is to the right of the centre of the screen.", lambda v: v > 0),
    },
}


def env_context(env_name):
    for k, v in ENV_CONTEXT.items():
        if k in env_name or env_name in k:
            return v
    return f"The environment is {env_name}. Each image shows one state."


def factor_tests(env_name):
    for k, v in FACTOR_TESTS.items():
        if k in env_name:
            return v
    return {}


class GeminiScorer:
    """
    API scorer (e.g. gemini-2.5-pro). Gemini gives no token log-probs for 2.5 Pro, so the
    model is asked for its probability (0-100) that the statement is true; this keeps AUC
    threshold-free. Key from env GEMINI_API_KEY. Temperature 0, small thinking budget.
    """
    PROB = ("\nReply with ONLY an integer from 0 to 100: your probability that the "
            "statement is true in this state.")

    def __init__(self, model_id, thinking_budget=128):
        from google import genai
        from google.genai import types
        self.model_id, self.types = model_id, types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_version = None                       # filled from the first API response
        self.errors = {}                                 # failure type -> count
        if model_id.startswith("gemini-3"):
            # Gemini 3.x: thinking_level only (thinking_budget -> 400), keep default temperature 1.0
            level = "HIGH" if thinking_budget >= 1024 else "LOW"
            self.config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=level))
            self.gen_params = dict(thinking_level=level, temperature="API default (1.0)")
        else:
            self.config = types.GenerateContentConfig(
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget))
            self.gen_params = dict(thinking_budget=thinking_budget, temperature=0.0)

    def _call(self, content, retries=8):
        import random
        parts = [c if not _is_image(c) else _load(c, 1024) for c in content]
        for k in range(retries):
            try:
                r = self.client.models.generate_content(model=self.model_id, contents=parts,
                                                        config=self.config)
                if self.model_version is None:
                    self.model_version = getattr(r, "model_version", None)
                if not r.text:                           # empty: blocked / cut off -> log reason
                    fr = getattr(r.candidates[0], "finish_reason", None) if r.candidates else None
                    self.errors[f"empty:{fr}"] = self.errors.get(f"empty:{fr}", 0) + 1
                return r.text or ""
            except Exception as e:                       # 429 / 5xx: back off and retry
                key = type(e).__name__ + ":" + str(getattr(e, "code", ""))
                if k == retries - 1:
                    self.errors[key] = self.errors.get(key, 0) + 1
                    print(f"  [gemini] giving up: {str(e)[:200]}")
                    return ""
                time.sleep(min(2 ** k, 60) + random.random() * 2)

    def prob(self, content):
        content = [c.replace("Answer with Yes or No only.", "").rstrip() if isinstance(c, str)
                   and not _is_image(c) else c for c in content]
        text = self._call(content + [self.PROB])
        m = re.search(r"\d{1,3}", text)
        if not m:
            return 0.5, 0.0                              # mass 0 = unusable answer
        return min(int(m.group(0)), 100) / 100.0, 1.0

    def chat(self, content):
        return self._call(content)


class BudgetExceeded(RuntimeError):
    pass


DEFAULT_PRICES = {            # USD per 1M tokens (input, output); override with --price_in/--price_out
    "gpt-5.5": (5.0, 30.0),
    "deepseek": (0.3, 1.5),   # not verified: check the DeepSeek dashboard and override
}


def default_price(model_id):
    for k, v in DEFAULT_PRICES.items():
        if model_id.startswith(k):
            return v
    return None


class _Ledger:
    """shared USD spend per model across processes (file + fcntl lock)"""

    def __init__(self, path, budget, price_in, price_out):
        self.path, self.budget, self.pin, self.pout = path, budget, price_in, price_out
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _update(self, add=0.0, calls=0):
        import fcntl
        with open(self.path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            try:
                d = json.loads(f.read() or "{}")
            except json.JSONDecodeError:            # never silently reset the spend record
                raise RuntimeError(f"corrupt spend ledger {self.path}; fix it by hand before continuing")
            d["usd"] = d.get("usd", 0.0) + add
            d["calls"] = d.get("calls", 0) + calls
            d.update(budget=self.budget, price_in=self.pin, price_out=self.pout)
            f.seek(0); f.truncate(); f.write(json.dumps(d))
            f.flush(); os.fsync(f.fileno())          # data must be on disk before the lock is released
            fcntl.flock(f, fcntl.LOCK_UN)
        return d["usd"]

    def check(self):
        spent = self._update()
        if spent >= self.budget:
            raise BudgetExceeded(f"budget {self.budget} USD reached ({spent:.2f} spent, {self.path})")

    def record(self, usage):
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self._update(pt * self.pin / 1e6 + ct * self.pout / 1e6, 1)


class DeepSeekScorer(GeminiScorer):
    """
    DeepSeek vision model via the OpenAI-compatible API (base_url https://api.deepseek.com),
    e.g. deepseek-flash (V4.1 Flash). Key from env DEEPSEEK_API_KEY. Images are sent inline as
    base64 PNG data URLs. Same probability prompt and failure handling as GeminiScorer.
    """

    def __init__(self, model_id, thinking_budget=None):
        from openai import OpenAI
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        self.model_version, self.errors, self.ledger = None, {}, None
        self.gen_params = dict(temperature=0.0, thinking="API default")

    def _create_kwargs(self):
        return dict(temperature=0.0)

    @staticmethod
    def _data_url(img):
        import base64, io
        buf = io.BytesIO(); _load(img, 1024).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _call(self, content, retries=8):
        import random
        parts = [{"type": "image_url", "image_url": {"url": self._data_url(c)}} if _is_image(c)
                 else {"type": "text", "text": c} for c in content]
        for k in range(retries):
            if self.ledger:
                self.ledger.check()                      # raises BudgetExceeded, stops the run
            try:
                r = self.client.chat.completions.create(
                    model=self.model_id, messages=[{"role": "user", "content": parts}],
                    **self._create_kwargs())
                if self.model_version is None:
                    self.model_version = getattr(r, "model", None)
                if self.ledger:
                    self.ledger.record(getattr(r, "usage", None))
                text = r.choices[0].message.content or ""
                if not text:
                    fr = r.choices[0].finish_reason
                    self.errors[f"empty:{fr}"] = self.errors.get(f"empty:{fr}", 0) + 1
                return text
            except Exception as e:
                code = getattr(e, "status_code", None)
                key = type(e).__name__ + ":" + str(code or "")
                if k == retries - 1 or code in (400, 401, 403, 404):   # 4xx: retrying will not help
                    self.errors[key] = self.errors.get(key, 0) + 1
                    print(f"  [{self.model_id}] giving up: {str(e)[:200]}")
                    return ""
                time.sleep(min(2 ** k, 60) + random.random() * 2)


class OpenAIModel(DeepSeekScorer):
    """
    OpenAI model (e.g. gpt-5.4-2026-03-05) via the chat completions API; key from env
    OPENAI_API_KEY. Reasoning models (gpt-5*, o*) take reasoning_effort instead of temperature.
    Used as namer (effort high) or scorer (effort low).
    """

    def __init__(self, model_id, thinking_budget=128):
        from openai import OpenAI
        self.model_id = model_id
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_version, self.errors, self.ledger = None, {}, None
        self.reasoning = model_id.startswith(("gpt-5", "o1", "o3", "o4"))
        self.effort = ("high" if thinking_budget >= 4096 else
                       "medium" if thinking_budget >= 1024 else "low")
        self.gen_params = (dict(reasoning_effort=self.effort, temperature="API default (reasoning model)")
                           if self.reasoning else dict(temperature=0.0))

    def _create_kwargs(self):
        if self.reasoning:
            return dict(reasoning_effort=self.effort, max_completion_tokens=32000)
        return dict(temperature=0.0)


def _is_image(c):
    return not (isinstance(c, str) and not c.lower().endswith((".png", ".jpg", ".jpeg")))


def attach_budget(model, budget_usd, price_in=None, price_out=None, ledger_dir="experiments/vlm"):
    """cap USD spend of an OpenAI-compatible API model (shared across processes)"""
    if budget_usd is None or not hasattr(model, "ledger"):
        return
    pin, pout = (price_in, price_out) if price_in is not None else (default_price(model.model_id) or (None, None))
    assert pin is not None and pout is not None, f"no price for {model.model_id}: pass --price_in/--price_out"
    model.ledger = _Ledger(os.path.join(ledger_dir, f"spend_{model.model_id}.json"), budget_usd, pin, pout)
    model.gen_params = dict(model.gen_params, budget_usd=budget_usd, price_in=pin, price_out=pout)
    print(f"[budget] {model.model_id}: {budget_usd} USD cap, prices {pin}/{pout} per 1M tokens, "
          f"ledger {model.ledger.path}")


def is_api(model):
    return isinstance(model, GeminiScorer)          # DeepSeekScorer is a subclass


def load_vlm(name_or_id, load_4bit=False, thinking_budget=128):
    """-> (model, processor). name_or_id: key of MODELS, HF repo id, local path, or gemini-*
    thinking_budget: Gemini only (128 = minimum for 2.5 Pro, enough for Yes/No scoring)"""
    if name_or_id.startswith("gemini"):
        return GeminiScorer(name_or_id, thinking_budget), None
    if name_or_id.startswith("deepseek"):
        return DeepSeekScorer(name_or_id), None
    if name_or_id.startswith(("gpt-", "o1", "o3", "o4")):
        return OpenAIModel(name_or_id, thinking_budget), None
    from transformers import AutoProcessor, AutoModelForImageTextToText
    model_id = MODELS.get(name_or_id, name_or_id)
    kw = dict(dtype=torch.bfloat16, device_map="auto")  # transformers >= 4.56 / 5.x
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    model.model_id = model_id
    model.gen_params = dict(dtype="bfloat16", load_4bit=load_4bit, decoding="greedy",
                            yes_no="P(Yes)/(P(Yes)+P(No)) at first answer token")
    return model, proc


def _versions():
    import platform
    from importlib.metadata import version, PackageNotFoundError
    out = {"python": platform.python_version()}
    for pkg in ["numpy", "torch", "transformers", "google-genai", "stable-baselines3",
                "gymnasium", "minigrid", "scikit-learn", "sentence-transformers", "captum"]:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def manifest(args, model=None):
    """reproducibility record stored in every output file"""
    import datetime, sys
    m = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             argv=sys.argv, args=vars(args), versions=_versions())
    if model is not None:
        m.update(model_id=model.model_id, generation=getattr(model, "gen_params", None),
                 api_model_version=getattr(model, "model_version", None))
    return m


def _load(image, max_side):
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)), Image.LANCZOS)
    return image


def _inputs(model, proc, content, max_side=1024):
    """content: list of str (text) and str-path / PIL (image), in order"""
    parts = [{"type": "image", "image": _load(c, max_side)} if _is_image(c)
             else {"type": "text", "text": c} for c in content]
    inputs = proc.apply_chat_template([{"role": "user", "content": parts}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")
    return inputs.to(model.device, dtype=torch.bfloat16)


@torch.no_grad()
def chat(model, proc, content, max_new_tokens=300):
    """greedy generation; content = [text | image, ...] -> str"""
    if is_api(model):
        return model.chat(content)
    inputs = _inputs(model, proc, content)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _first_ids(tok, words):
    ids = set()
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if t:
            ids.add(t[0])
    return sorted(ids)


@torch.no_grad()
def yes_prob(model, proc, content):
    """
    -> (p_yes, mass). p_yes = P(Yes) / (P(Yes) + P(No)) at the first answer token;
    mass = P(Yes) + P(No); low mass means the model is not answering Yes/No.
    API models: stated probability, mass 1 (0 if the answer could not be parsed).
    """
    if is_api(model):
        return model.prob(content)
    tok = proc.tokenizer
    if not hasattr(model, "_yn_ids"):
        model._yn_ids = (_first_ids(tok, ["Yes", "yes", " Yes", " yes"]),
                         _first_ids(tok, ["No", "no", " No", " no"]))
    y_ids, n_ids = model._yn_ids
    logits = model(**_inputs(model, proc, content)).logits[0, -1].float()
    lp = logits.log_softmax(-1)
    ly, ln = torch.logsumexp(lp[y_ids], 0), torch.logsumexp(lp[n_ids], 0)
    return torch.sigmoid(ly - ln).item(), (ly.exp() + ln.exp()).item()


def parse_json(text):
    """LAST {...} block containing "label" -> dict, else {}"""
    for block in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            d = json.loads(block)
            if "label" in d:
                return d
        except json.JSONDecodeError:
            continue
    m = re.findall(r'"label"\s*:\s*"([^"]+)"', text)
    return {"label": m[-1]} if m else {}


def label_to_text(label):
    return label.replace("_", " ").strip()
