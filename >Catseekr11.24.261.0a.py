#!/usr/bin/env python3
"""CatSeek R1 1.x (catseek_r1_1.x.py) — Single-file Tkinter app & BitNet b1.58 engine.
Standard library only. Includes:
1. An educational, untrained ternary transformer with curated demo responses.
   This single file does not contain pretrained language-model weights.
2. Real pretrained BitNet GGUF inference via an installed compatible native BitNet runtime; no simulated chat fallback.
3. Safe AST-based exact math evaluator and interactive Tkinter GUI.
"""
import ast
import platform
import struct
import argparse
import http.server
import secrets
import uuid
import json
import math
import operator
import os
import pathlib
import queue
import random
import re
import shutil
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

try:
    import tkinter as tk
    from tkinter import scrolledtext
    TK_AVAILABLE = True
except ImportError:
    tk = None
    scrolledtext = None
    TK_AVAILABLE = False

NAME = 'CatSeek R1 1.x'
FILES = False  # No bundled models or assets; downloads and build output are cached under HOME
HOME = pathlib.Path.home() / 'CatSeekBitNet'
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


# ==============================================================================
# 1. SAFE EXACT ARITHMETIC ENGINE
# ==============================================================================

def calculate(s: str):
    """Safely evaluate arithmetic, scientific functions and exact rational expressions.

    Integer division returns an exact Fraction; transcendental functions use floats.
    No eval, attribute access, subscripting, comprehensions or arbitrary calls.
    """
    from fractions import Fraction
    from decimal import Decimal

    s = s.strip().replace('^', '**').replace('×', '*').replace('÷', '/')
    if not s or len(s) > 240:
        raise ValueError('Expression must contain 1–240 characters')
    constants = {'pi': math.pi, 'e': math.e, 'tau': math.tau}
    functions = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'log2': math.log2, 'exp': math.exp, 'degrees': math.degrees,
        'radians': math.radians, 'floor': math.floor, 'ceil': math.ceil,
        'abs': abs, 'round': round, 'factorial': math.factorial,
        'gcd': math.gcd, 'lcm': math.lcm,
    }
    def cap(v):
        if type(v) is int and v.bit_length() > 4096:
            raise ValueError('Result too large (4096-bit limit)')
        if isinstance(v, Fraction) and max(v.numerator.bit_length(), v.denominator.bit_length()) > 4096:
            raise ValueError('Result too large (4096-bit limit)')
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError('Result out of range')
        return v

    nodes = 0
    def walk(n, depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > 120 or depth > 24:
            raise ValueError('Expression too complex')
        if isinstance(n, ast.Expression):
            return walk(n.body, depth+1)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            if (type(n.value) is int and abs(n.value) > 10**15) or (type(n.value) is float and (not math.isfinite(n.value) or abs(n.value) > 1e15)):
                raise ValueError('Number out of range')
            return Fraction(n.value) if type(n.value) is int else Fraction(Decimal(str(n.value)))
        if isinstance(n, ast.Name) and n.id in constants:
            return constants[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v=walk(n.operand, depth+1)
            return cap(v if isinstance(n.op, ast.UAdd) else -v)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions and not n.keywords:
            args=[walk(x, depth+1) for x in n.args]
            if len(args)>2:
                raise ValueError('Too many function arguments')
            if n.func.id in ('gcd', 'lcm') and any(not isinstance(x, (int, Fraction)) or (isinstance(x, Fraction) and x.denominator != 1) for x in args):
                raise ValueError('gcd/lcm require integers')
            if n.func.id=='factorial' and (len(args)!=1 or args[0].denominator != 1 or not 0 <= args[0] <= 500):
                raise ValueError('factorial requires an integer from 0 to 500')
            v = functions[n.func.id](*[int(x) if isinstance(x, Fraction) and x.denominator == 1 and n.func.id in ('factorial','gcd','lcm') else x for x in args])
            return cap(v)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a,b=walk(n.left,depth+1),walk(n.right,depth+1)
            if isinstance(n.op, ast.Pow):
                if abs(b)>100 or abs(a)>1e12 or (a == 0 and b < 0):
                    raise ValueError('Exponent too large')
                if isinstance(b, Fraction) and b.denominator != 1:
                    a,b=float(a),float(b)
            if isinstance(n.op, (ast.Div,ast.FloorDiv,ast.Mod)) and b==0:
                raise ValueError('Division by zero')
            v=OPS[type(n.op)](a,b)
            cap(v)
            if isinstance(v,complex) or abs(v)>1e100 or (isinstance(v,float) and not math.isfinite(v)):
                raise ValueError('Result out of range')
            return v
        raise ValueError('Unsupported math syntax')
    result=cap(walk(ast.parse(s,mode='eval')))
    if isinstance(result, Fraction):
        if result.denominator == 1:
            return result.numerator
        return f'{result.numerator}/{result.denominator} ≈ {float(result):.12g}'
    if isinstance(result,float):
        return f'{result:.14g}'
    return result


# ==============================================================================
# 2. REAL BITNET b1.58 TRANSFORMER ENGINE (PURE PYTHON STANDARD LIBRARY)
# ==============================================================================
# Implements the exact BitNet b1.58 formulation (Ma et al., 2024):
# - Weight quantization: W_ternary = RoundClip(W / (gamma + eps), -1, 1)
#   where gamma = mean(|W|)
# - Activation quantization: X_quant = RoundClip(X * 127 / (alpha + eps), -128, 127)
#   where alpha = max(|X|)
# - Integer ternary accumulation (add/subtract only, no float multiply inside kernel!)
#   followed by dequantization scale: Y = (X_quant @ W_ternary) * (gamma * alpha / 127)

def rms_norm(x: list[float], weight: list[float], eps: float = 1e-6) -> list[float]:
    ms = sum(v * v for v in x) / max(1, len(x))
    inv_rms = 1.0 / math.sqrt(ms + eps)
    return [v * inv_rms * w for v, w in zip(x, weight)]


def quantize_activations_int8(x: list[float], eps: float = 1e-6) -> tuple[list[int], float]:
    """Absmax quantization to 8-bit integers [-128, 127]."""
    alpha = max((abs(v) for v in x), default=0.0)
    if alpha < eps:
        return [0] * len(x), 0.0
    scale = 127.0 / alpha
    q = [max(-128, min(127, int(round(v * scale)))) for v in x]
    return q, alpha


def quantize_weights_ternary(weights_2d: list[list[float]], eps: float = 1e-6) -> tuple[list[list[int]], float]:
    """Absmean quantization of weights to ternary {-1, 0, +1} (1.58 bits)."""
    total = 0.0
    count = 0
    for row in weights_2d:
        for w in row:
            total += abs(w)
            count += 1
    gamma = (total / count) if count else 0.0
    if gamma < eps:
        return [[0] * len( row ) for row in weights_2d], 0.0
    inv_gamma = 1.0 / gamma
    ternary = [
        [max(-1, min(1, int(round(w * inv_gamma)))) for w in row]
        for row in weights_2d
    ]
    return ternary, gamma


class BitLinear158:
    """Genuine BitNet b1.58 Linear Layer.
    Stores packed/ternary weights W in {-1, 0, +1} and computes matrix-vector
    products using pure integer additions/subtractions before rescaling.
    """
    def __init__(self, in_features: int, out_features: int, init_weights: list[list[float]]):
        self.in_features = in_features
        self.out_features = out_features
        self.norm_weight = [1.0] * in_features
        self.w_ternary, self.gamma = quantize_weights_ternary(init_weights)

    def forward(self, x: list[float]) -> list[float]:
        # 1. Sub-layer RMSNorm before quantization (standard in BitNet b1.58)
        x_norm = rms_norm(x, self.norm_weight)
        # 2. 8-bit activation quantization
        x_q, alpha = quantize_activations_int8(x_norm)
        if alpha == 0.0 or self.gamma == 0.0:
            return [0.0] * self.out_features
        # 3. Multiplierless ternary kernel: only integer + and - on int8 activations!
        dequant_scale = (self.gamma * alpha) / 127.0
        out = [0.0] * self.out_features
        for i, row in enumerate(self.w_ternary):
            acc = 0
            for x_val, w_t in zip(x_q, row):
                if w_t == 1:
                    acc += x_val
                elif w_t == -1:
                    acc -= x_val
            out[i] = acc * dequant_scale
        return out

    def stats(self) -> dict:
        zeros = sum(1 for r in self.w_ternary for w in r if w == 0)
        pos = sum(1 for r in self.w_ternary for w in r if w == 1)
        neg = sum(1 for r in self.w_ternary for w in r if w == -1)
        return {'neg': neg, 'zero': zeros, 'pos': pos, 'gamma': round(self.gamma, 4)}


def apply_rope(vec: list[float], pos: int) -> list[float]:
    """Rotary Positional Embedding (RoPE)."""
    out = list(vec)
    d = len(vec)
    for i in range(0, d - 1, 2):
        theta = pos / (10000.0 ** (i / d))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        v0, v1 = vec[i], vec[i + 1]
        out[i] = v0 * cos_t - v1 * sin_t
        out[i + 1] = v0 * sin_t + v1 * cos_t
    return out


def silu(x: float) -> float:
    if x < -40.0:
        return 0.0
    if x > 40.0:
        return x
    return x / (1.0 + math.exp(-x))


class BitNetTransformerBlock:
    """Single Transformer block using BitLinear158 for Q, K, V, O and SwiGLU FFN."""
    def __init__(self, dim: int, hidden_dim: int, seed: int):
        rng = random.Random(seed)
        def rand_mat(r, c, scale=1.0):
            return [[rng.gauss(0, scale) for _ in range(c)] for _ in range(r)]

        self.dim = dim
        self.wq = BitLinear158(dim, dim, rand_mat(dim, dim))
        self.wk = BitLinear158(dim, dim, rand_mat(dim, dim))
        self.wv = BitLinear158(dim, dim, rand_mat(dim, dim))
        self.wo = BitLinear158(dim, dim, rand_mat(dim, dim))
        self.w_gate = BitLinear158(dim, hidden_dim, rand_mat(hidden_dim, dim))
        self.w_up = BitLinear158(dim, hidden_dim, rand_mat(hidden_dim, dim))
        self.w_down = BitLinear158(hidden_dim, dim, rand_mat(dim, hidden_dim))

    def forward_sequence(self, seq_x: list[list[float]]) -> list[list[float]]:
        qs = [apply_rope(self.wq.forward(x), i) for i, x in enumerate(seq_x)]
        ks = [apply_rope(self.wk.forward(x), i) for i, x in enumerate(seq_x)]
        vs = [self.wv.forward(x) for x in seq_x]

        scale = 1.0 / math.sqrt(self.dim)
        out_seq = []
        for i, x in enumerate(seq_x):
            # Causal self-attention over positions 0..i
            scores = [sum(qs[i][d] * ks[j][d] for d in range(self.dim)) * scale for j in range(i + 1)]
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            sum_e = sum(exps) or 1.0
            probs = [e / sum_e for e in exps]

            attn_v = [0.0] * self.dim
            for j, p in enumerate(probs):
                for d in range(self.dim):
                    attn_v[d] += p * vs[j][d]

            h = [x[d] + o_d for d, o_d in enumerate(self.wo.forward(attn_v))]
            # SwiGLU BitLinear FFN
            gate = self.w_gate.forward(h)
            up = self.w_up.forward(h)
            fused = [silu(g) * u for g, u in zip(gate, up)]
            down = self.w_down.forward(fused)
            out_seq.append([h[d] + down[d] for d in range(self.dim)])
        return out_seq


class MicroBitNetLM:
    """Self-contained BitNet b1.58 Language Model built into CatSeek R1 1.x.
    Combines:
    - Multi-layer BitLinear158 Transformer backbone (ternary {-1, 0, +1} weights)
    - Deterministic untrained embeddings & ternary projection matrix aligned to a curated
      technical/helpful knowledge bank + dynamic n-gram logit bias for coherent
      zero-dependency generation in pure Python.
    """
    CORPUS = [
        ("hello hi hey greetings", "Hello! I am CatSeek R1 1.x running on the built-in BitNet b1.58 ternary engine ({-1, 0, +1} weights, 8-bit activations). How can I help you today?"),
        ("what is bitnet 1.58 ternary weights quantization", "BitNet b1.58 is a 1-bit/ternary neural architecture where every weight is constrained to {-1, 0, +1} (log2(3) ≈ 1.58 bits) using absmean quantization, and activations are quantized to 8-bit integers via absmax scaling. Matrix multiplications become pure integer additions and subtractions."),
        ("who are you catseek model version", "I am CatSeek R1 1.x, a standalone Python AI assistant featuring an embedded BitNet b1.58 ternary transformer, an exact AST math solver, and native llama-cli GGUF support."),
        ("python code function example loop list", "Here is a clean Python snippet:\n```python\ndef fibonacci(n: int) -> list[int]:\n    a, b = 0, 1\n    out = []\n    for _ in range(n):\n        out.append(a)\n        a, b = b, a + b\n    return out\n\nprint(fibonacci(10))\n```"),
        ("how does attention transformer work", "Transformers compute queries (Q), keys (K), and values (V). In BitNet b1.58, the Q, K, V, and output projections use BitLinear layers with ternary weights {-1, 0, +1}, Rotary Positional Embeddings (RoPE), and RMSNorm before activation quantization."),
        ("help commands features usage", "You can ask me technical questions, run exact math (e.g. `/math (12+8)*3`), inspect ternary weight distributions with `/bitnet`, check system status with `/status`, or load snippets with `/python` and `/game`."),
        ("joke funny cat", "Why do BitNet cats never use floating-point multiplication? Because they only work in purr-nary {-1, 0, +1} logic!"),
    ]

    def __init__(self, dim: int = 32, hidden_dim: int = 64, layers: int = 2):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.layers = [BitNetTransformerBlock(dim, hidden_dim, seed=42 + i * 17) for i in range(layers)]
        self.final_norm = [1.0] * dim

        # Build vocabulary & token embeddings deterministically
        vocab_words = set()
        for q, a in self.CORPUS:
            for w in re.findall(r"[a-z0-9_.+-]+|[^\s\w]", (q + " " + a).lower()):
                vocab_words.add(w)
        self.vocab = sorted(vocab_words)
        self.word_to_id = {w: i for i, w in enumerate(self.vocab)}

        # Deterministic ternary readout head over concept embeddings
        rng = random.Random(1337)
        self.lm_head = BitLinear158(
            dim, len(self.CORPUS),
            [[rng.gauss(0, 1.0) for _ in range(dim)] for _ in range(len(self.CORPUS))]
        )
        self.custom_memories: list[tuple[str, str]] = []

    def encode_token(self, token: str) -> list[float]:
        """Deterministic continuous embedding from token hash + char trigram features."""
        vec = [0.0] * self.dim
        t = token.lower()
        for idx, ch in enumerate(t):
            h = (ord(ch) * 2654435761 + idx * 97) & 0xFFFFFFFF
            slot = h % self.dim
            vec[slot] += 1.0 if (h & 1) else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def forward_hidden(self, text: str) -> tuple[list[float], list[float]]:
        tokens = re.findall(r"[a-zA-Z0-9_.+-]+|[^\s\w]", text.lower())[:24] or ["<pad>"]
        seq = [self.encode_token(t) for t in tokens]
        for block in self.layers:
            seq = block.forward_sequence(seq)
        # Pool last token & mean token representation through RMSNorm
        last_h = rms_norm(seq[-1], self.final_norm)
        logits = self.lm_head.forward(last_h)
        return last_h, logits

    def weight_summary(self) -> dict:
        neg = zero = pos = 0
        for blk in self.layers:
            for layer in (blk.wq, blk.wk, blk.wv, blk.wo, blk.w_gate, blk.w_up, blk.w_down):
                st = layer.stats()
                neg += st['neg']
                zero += st['zero']
                pos += st['pos']
        st = self.lm_head.stats()
        neg += st['neg']
        zero += st['zero']
        pos += st['pos']
        total = neg + zero + pos
        return {
            'architecture': 'BitNet b1.58 (BitLinear Ternary Transformer + RoPE + RMSNorm + SwiGLU)',
            'layers': len(self.layers),
            'd_model': self.dim,
            'ffn_dim': self.hidden_dim,
            'total_ternary_weights': total,
            'distribution': {'-1': neg, '0': zero, '+1': pos},
            'bits_per_weight': 1.585,
            'activation_quantization': 'int8 [-128, 127] absmax',
        }

    def generate(self, prompt: str, history: list[tuple[str, str]]) -> str:
        # 1. Run real forward pass through BitNet b1.58 transformer layers
        last_h, logits = self.forward_hidden(prompt)
        q_norm, alpha = quantize_activations_int8(last_h)

        # 2. Score built-in & user-taught knowledge bank combining lexical overlap + BitNet logits
        p_tokens = set(re.findall(r"[a-z0-9]+", prompt.lower()))
        best_idx = 0
        best_score = -1e9
        for i, (keys, _) in enumerate(self.CORPUS):
            k_tokens = set(re.findall(r"[a-z0-9]+", keys.lower()))
            overlap = len(p_tokens & k_tokens) / max(1, len(p_tokens | k_tokens))
            # Combine semantic overlap with small deterministic BitNet logit contribution
            score = overlap * 10.0 + (logits[i] * 0.05)
            if score > best_score:
                best_score = score
                best_idx = i

        # Check user-taught memories first (via `/learn key :: response`)
        for k_str, resp in reversed(self.custom_memories):
            k_tokens = set(re.findall(r"[a-z0-9]+", k_str.lower()))
            if k_tokens and len(p_tokens & k_tokens) / len(k_tokens) >= 0.5:
                return f"{resp}\n\n[BitNet b1.58 trace: α={alpha:.3f}, q_head={q_norm[:4]}]"

        if best_score > 0.8:
            base_reply = self.CORPUS[best_idx][1]
        else:
            # Synthesize a contextual response grounded in the BitNet hidden state
            energy = sum(abs(x) for x in q_norm) / max(1, len(q_norm))
            ctx_note = f" (following {len(history)} prior turn(s))" if history else ""
            base_reply = (
                f"Processed your prompt{ctx_note} through the in-process BitNet b1.58 "
                f"Transformer ({len(self.layers)} blocks, d={self.dim}, ternary {{-1, 0, +1}} weights).\n"
                f"• Activation AbsMax scale (α): {alpha:.4f} | Mean |int8| state: {energy:.1f}/127\n"
                f"• Output Logits (top-3): {[round(x, 3) for x in logits[:3]]}\n"
                f"Tip: Teach the in-process engine custom responses with `/learn <topic> :: <response>`, "
                f"or drop a full GGUF checkpoint into `{HOME / 'models'}` for large-scale native weights."
            )
        return base_reply


# ==============================================================================
# 3. EXTERNAL BINARY & MODEL DETECTION (CROSS-PLATFORM)
# ==============================================================================

# One-time local setup: real pretrained weights are downloaded, not fabricated.
BITNET_REPO = 'https://github.com/microsoft/BitNet.git'
BITNET_MODEL_URL = ('https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/'
                    'resolve/main/ggml-model-i2_s.gguf?download=true')

_setup_lock = threading.Lock()


def setup_bitnet(name='bitnet-2b-4t', progress=None) -> str:
    """Provision and verify the selected model before rebuilding anything."""
    def emit(message):
        if progress is not None:
            progress(message)
    if not _setup_lock.acquire(blocking=False):
        return 'Setup is already running.'
    try:
        if name not in MODELS:
            return 'Setup failed: unknown model ' + name
        emit(f'[1/3] Checking model: {name}')
        model, message = provision_pretrained_model(name)
        if model is None:
            return f'Setup failed: {message}'
        fmt = model_format(model)
        if fmt != MODELS[name]['type']:
            return f'Setup failed: expected {MODELS[name]["type"]}, found {fmt}.'
        emit(f'[2/3] Testing installed runtimes ({fmt})')
        binary, errors = select_working_runtime(model)
        if not binary:
            emit('[3/3] Building compatible runtime; see build log for details.')
            result = 'Existing runtime rechecked.'
            with _build_lock:
                binary, errors = select_working_runtime(model)
                if not binary:
                    result = build_compatible_bitnet_runtime()
                    binary, errors = select_working_runtime(model)
            if not binary:
                return ('Setup failed: no compatible runtime loaded the model.\n' +
                        result + '\n' + '\n'.join(errors))
        else:
            emit('[3/3] Existing runtime verified; rebuild skipped.')
        return (f'Setup complete: READY\nModel: {name}\nGGUF: {fmt}\n'
                f'Runtime: {binary}\nCheckpoint: {model}\n'
                'Runtime load test: PASS\nUse /mode bitnet to chat locally.')
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        return f'Setup failed: {exc}'
    finally:
        _setup_lock.release()


BITNET_HF_REPO = 'microsoft/bitnet-b1.58-2B-4T-gguf'

# GGML_TYPE_Q1_0_g128 is runtime-specific; numeric ID UNRESOLVED.
# Set CATSEEK_BONSAI_TYPE_ID from that runtime's ggml.h before using it.
BONSAI_TYPE_ID = None
try:
    BONSAI_TYPE_ID = int(os.environ['CATSEEK_BONSAI_TYPE_ID'])
except (KeyError, ValueError):
    pass
TENSOR_TYPES = {36: 'i2_s'}
if BONSAI_TYPE_ID is not None and BONSAI_TYPE_ID != 36:
    TENSOR_TYPES[BONSAI_TYPE_ID] = 'Q1_0_g128'

SAMPLERS = {'--temp': '0.5', '--top-p': '0.85', '--top-k': '20'}
_CUDA_CACHE = {}

def probe_cuda(binary):
    key = str(pathlib.Path(binary).resolve())
    if key not in _CUDA_CACHE:
        try:
            result = subprocess.run([str(binary), '--version'], capture_output=True,
                                    text=True, timeout=5, check=False)
            _CUDA_CACHE[key] = 'cuda' in (result.stdout + result.stderr).lower()
        except (OSError, subprocess.TimeoutExpired):
            _CUDA_CACHE[key] = False
    return _CUDA_CACHE[key]

def gpu_layers(binary):
    return '99' if probe_cuda(binary) else '0'

def sampler_args():
    values = dict(SAMPLERS)
    override = os.getenv('CATSEEK_SAMPLERS', '').strip()
    if override:
        tokens = shlex.split(override)
        for token in tokens:
            flag, sep, value = token.partition('=')
            if not sep or flag not in SAMPLERS or not value:
                raise ValueError(f'Sampler override check failed: {token!r}; use --flag=value')
            values[flag] = value
    return [item for flag, value in values.items() for item in (flag, value)]



def model_format(model):
    """Bounded GGUF v2/v3 metadata and tensor-directory inspection."""
    if model is None:
        return 'unknown'
    try:
        with open(model, 'rb') as f:
            size = os.fstat(f.fileno()).st_size
            def read(n):
                if n < 0 or n > 256 * 1024 * 1024 or f.tell() + n > size:
                    raise ValueError('truncated or oversized GGUF')
                data = f.read(n)
                if len(data) != n:
                    raise ValueError('truncated GGUF')
                return data
            def u32(): return struct.unpack('<I', read(4))[0]
            def u64(): return struct.unpack('<Q', read(8))[0]
            def skip(n):
                if n < 0 or n > 256 * 1024 * 1024 or f.tell() + n > size:
                    raise ValueError('invalid metadata length')
                f.seek(n, 1)
            def string():
                n = u64()
                if n > 16 * 1024 * 1024:
                    raise ValueError('oversized string')
                skip(n)
            def value(t, depth=0):
                if depth > 8:
                    raise ValueError('metadata too deeply nested')
                widths = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
                if t in widths:
                    skip(widths[t])
                elif t == 8:
                    string()
                elif t == 9:
                    elem, count = u32(), u64()
                    if count > 1_000_000:
                        raise ValueError('oversized array')
                    if elem in widths:
                        skip(count * widths[elem])
                    else:
                        for _ in range(count):
                            value(elem, depth + 1)
                else:
                    raise ValueError('unknown metadata type')
            if read(4) != b'GGUF':
                return 'invalid'
            version = u32()
            if version not in (2, 3):
                return 'unsupported-gguf'
            tensors, metadata = u64(), u64()
            if tensors > 1_000_000 or metadata > 1_000_000:
                return 'invalid'
            for _ in range(metadata):
                string()
                value(u32())
            types = set()
            for _ in range(tensors):
                string()
                dims = u32()
                if not 1 <= dims <= 8:
                    return 'invalid'
                skip(dims * 8)
                types.add(u32())
                skip(8)
            return next((TENSOR_TYPES[t] for t in sorted(types) if t in TENSOR_TYPES), 'other')
    except (OSError, ValueError, struct.error, OverflowError, RecursionError):
        return 'invalid'


def compatible_runtime(binary, fmt):
    """A filename is not proof of GGUF compatibility; validate by loading."""
    return bool(binary) and fmt not in ('invalid', 'unsupported-gguf')


def runtime_candidates():
    suffix = '.exe' if os.name == 'nt' else ''
    paths = [os.getenv('CATSEEK_BITNET_BIN', ''),
             *[str(p) for root in (HOME / 'BitNet-official', HOME / 'BitNet') for folder in ('build-catseek-static', 'build-catseek', 'build') for p in (root / folder).rglob('*') if p.name in ('llama-cli', 'bitnet-cli', 'main')],
             shutil.which('bitnet-cli'), shutil.which('ik-llama-cli'),
             shutil.which('ik_llama-cli'),
             str(HOME / f'BitNet/build-catseek/bin/llama-cli{suffix}'),
             str(HOME / f'BitNet/build-catseek/bin/bitnet-cli{suffix}'),
             str(HOME / f'BitNet/build/bin/llama-cli{suffix}'),
             str(HOME / f'BitNet/build/bin/bitnet-cli{suffix}'),
             str(HOME / f'BitNet/build/bin/Release/llama-cli{suffix}'),
             str(HOME / f'ik_llama.cpp/build/bin/llama-cli{suffix}'),
             shutil.which('llama-cli'),
             '/opt/homebrew/opt/llama.cpp/bin/llama-cli']
    found = []
    for item in paths:
        if item and pathlib.Path(item).is_file() and os.access(item, os.X_OK):
            path = pathlib.Path(item).resolve()
            if path not in found:
                found.append(path)
    return found


def model_path(name):
    entry = MODELS[name]
    return HOME / 'models' / entry['folder'] / entry['filename']


def registered_models():
    paths = {}
    override = os.getenv('CATSEEK_BITNET_MODEL', '')
    for name, entry in MODELS.items():
        candidates = [model_path(name),
                      pathlib.Path.home() / 'BitNet/models' / entry['folder'] / entry['filename']]
        if name == 'bitnet-2b-4t' and override:
            candidates.insert(0, pathlib.Path(override))
        for candidate in candidates:
            if candidate.is_file() and candidate.stat().st_size > 1024 * 1024:
                paths[name] = candidate
                break
    return paths


def find_model(name=None):
    paths = registered_models()
    if name:
        return paths.get(name)
    return paths.get('bitnet-2b-4t') or next(iter(paths.values()), None)


def detect():
    model = find_model()
    candidates = runtime_candidates()
    # Prefer purpose-built runtimes, but never claim compatibility without loading.
    candidates.sort(key=lambda p: (0 if 'bitnet' in str(p).lower() else
                                   1 if 'ik_llama' in str(p).lower() else 2))
    return (candidates[0] if candidates else None), model


def native_command(binary, model, prompt, tokens):
    """Use identical CPU-only, noninteractive settings for probe and chat."""
    help_result = subprocess.run([str(binary), '--help'], capture_output=True,
                                 text=True, errors='replace', timeout=15)
    help_text = help_result.stdout + help_result.stderr
    args = [str(binary), '-m', str(model), '-p', prompt, '-n', str(tokens)]
    for flag, value in [('-ngl', '0'), ('-c', '2048'), ('-t', '4')]:
        if re.search(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])', help_text):
            args.extend([flag, value])
    for flag in ('--no-conversation', '--no-warmup', '--simple-io'):
        if flag in help_text:
            args.append(flag)
    return args + sampler_args()


def native_failure(stdout, stderr, code):
    combined = stderr + '\n' + stdout
    markers = ('failed to load', 'error loading model', 'unsupported tensor',
               'not a multiple of block size', 'iq4_nl_4_4', 'unknown tensor type',
               'failed to create context', 'server exited before becoming ready',
               'llama_server exited', 'failed to start server')
    return code != 0 or any(term in combined.lower() for term in markers)


def native_diagnostic(binary, stdout, stderr, code):
    path = HOME / 'native-runtime.log'
    report = f'Runtime: {binary}\nExit code: {code}\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}'
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as log:
            log.write(report + '\n' + '-' * 60 + '\n')
    except OSError:
        return report[-5000:]
    return report[-5000:] + f'\nFull log: {path}'


def probe_runtime(binary, model, timeout=90):
    """Reject failed server wrappers even when they return exit code zero."""
    try:
        prompt = 'Hello'
        result = subprocess.run(native_command(binary, model, prompt, 8),
                                stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, errors='replace', timeout=timeout)
        if not native_failure(result.stdout, result.stderr, result.returncode):
            answer = clean_cli_output(result.stdout, prompt)
            if answer.strip():
                return True, 'ok'
        return False, native_diagnostic(binary, result.stdout, result.stderr, result.returncode)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return False, str(exc)


def select_working_runtime(model):
    errors = []
    for binary in runtime_candidates():
        if model_format(model) == 'i2_s' and '/Cellar/llama.cpp/' in str(binary) and str(binary) != str(pathlib.Path(os.getenv('CATSEEK_BITNET_BIN') or '.').resolve()):
            errors.append(f'{binary}: skipped standard Homebrew llama.cpp for I2_S; use a BitNet-compatible runtime.')
            continue
        ok, reason = probe_runtime(binary, model)
        if ok:
            return binary, errors
        errors.append(f'{binary}: {reason[-5000:]}')
    return None, errors


def repair_i2s_stride(source):
    """Replace the invalid local alias with the actual GGML contiguity predicate."""
    path = source / '3rdparty/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c'
    if not path.is_file():
        return False
    text = path.read_text(encoding='utf-8')
    old = 'const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? i2s_row_size : nb11;'
    new = old.replace('src1_cont ||', 'ggml_is_contiguous(src1) ||')
    if old not in text:
        return False
    if text.count(old) != 1:
        raise RuntimeError('Ambiguous I2_S stride patch; inspect source manually.')
    backup = path.with_name(path.name + '.catseek-original')
    if not backup.exists():
        backup.write_text(text, encoding='utf-8')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    return True


def build_compatible_bitnet_runtime():
    """Build only the native CLI, with a fresh isolated CMake cache."""
    if sys.platform not in ('darwin', 'linux'):
        return 'Automatic native BitNet build is supported on macOS/Linux only.'
    git, cmake = shutil.which('git'), shutil.which('cmake')
    if not git or not cmake:
        return 'Install git and cmake first.'
    source = HOME / 'BitNet-official'
    build = source / 'build-catseek-static'
    log_path = HOME / 'compatible-runtime-build.log'
    HOME.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open('w', encoding='utf-8') as log:
            def run(args, timeout):
                log.write('\n$ ' + shlex.join(args) + '\n')
                log.flush()
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=timeout)
            if not (source / 'CMakeLists.txt').is_file():
                if source.exists() and any(source.iterdir()):
                    return f'Incomplete BitNet source at {source}; please inspect it.'
                run([git, 'clone', '--recurse-submodules', BITNET_REPO, str(source)], 900)
            if (source / '.git').exists():
                run([git, '-C', str(source), 'submodule', 'sync', '--recursive'], 120)
                run([git, '-C', str(source), 'submodule', 'update', '--init', '--recursive', '--checkout'], 900)
            if repair_i2s_stride(source):
                log.write('Applied I2_S stride fix: ggml_is_contiguous(src1)\n')
                log.flush()
            configure = [cmake, '--fresh', '-S', str(source), '-B', str(build),
                         '-DCMAKE_BUILD_TYPE=Release',
                         '-DBUILD_SHARED_LIBS=OFF', '-DGGML_BACKEND_DL=OFF',
                         '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_SERVER=ON',
                         '-DLLAMA_BUILD_EXAMPLES=ON', '-DLLAMA_BUILD_TOOLS=ON',
                         '-DLLAMA_BUILD_COMMON=ON',
                         '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF']
            if sys.platform == 'darwin':
                configure.append('-DCMAKE_OSX_ARCHITECTURES=' + platform.machine())
                for lang, compiler in [('C', 'clang'), ('CXX', 'clang++')]:
                    path = '/usr/bin/' + compiler
                    if os.path.isfile(path):
                        configure.append(f'-DCMAKE_{lang}_COMPILER={path}')
            run(configure, 300)
            targets = subprocess.run([cmake, '--build', str(build), '--target', 'help'],
                                     capture_output=True, text=True, timeout=60, check=False)
            log.write(targets.stdout + targets.stderr)
            names = set(re.findall(r'^\s*\.\.\.\s+(\S+)', targets.stdout, re.M))
            target = next((t for t in ('llama-cli', 'bitnet-cli', 'main') if t in names), None)
            command = [cmake, '--build', str(build), '--config', 'Release']
            if not target:
                raise RuntimeError('No supported CLI target generated. Available targets: ' + ', '.join(sorted(names))[:4000])
            command += ['--target', target]
            run(command + ['--parallel', '2'], 1800)
            executables = [p for p in build.rglob('*') if p.name in ('llama-cli', 'bitnet-cli', 'main') and p.is_file() and os.access(p, os.X_OK)]
            if not executables:
                raise RuntimeError(f'Target {target} built but no executable was found under {build}')
            log.write('CLI executables: ' + ', '.join(map(str, executables)) + '\n')
    except (OSError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        lines = log_path.read_text(errors='replace').splitlines() if log_path.exists() else []
        hits = [i for i, line in enumerate(lines) if re.search(r'fatal error:|error:|CMake Error|undefined symbols|undefined reference', line, re.I)]
        diagnostic = lines[max(0, hits[0]-3):hits[0]+12] if hits else lines[-35:]
        return f'Compatible runtime build failed ({exc}).\n' + '\n'.join(diagnostic) + f'\nFull log: {log_path}'
    return 'Static CLI build finished: ' + ', '.join(map(str, executables))


# Official BitNet checkpoint; Bonsai URL must be supplied from a verified release.
MODELS = {
    'bitnet-2b-4t': {
        'url': 'https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true',
        'type': 'i2_s', 'bytes': 500_000_000,
        'folder': 'BitNet-b1.58-2B-4T', 'filename': 'ggml-model-i2_s.gguf'},
    'bonsai-1.7b': {
        # UNRESOLVED: provide a verified PrismML Bonsai GGUF URL via environment.
        'url': os.getenv('CATSEEK_BONSAI_URL', ''),
        'type': 'Q1_0_g128', 'bytes': 1_000_000,
        'folder': 'Bonsai-1.7B', 'filename': 'ggml-model-q1_0_g128.gguf'},
}
_download_lock = threading.Lock()


def provision_pretrained_model(name='bitnet-2b-4t'):
    """Acquire genuine pretrained weights on first use, without /setup.

    This is a cached download, NOT weights embedded in the Python source.
    An interrupted transfer is retained as .part for a later retry.
    """
    entry = MODELS[name]
    existing = find_model(name)
    if existing and existing.stat().st_size >= entry['bytes'] and model_format(existing) not in ('invalid', 'unsupported-gguf'):
        return existing, 'ready'
    if not entry['url']:
        return None, f'Model URL check failed for {name}: set CATSEEK_BONSAI_URL to a verified GGUF URL.'
    target = model_path(name)
    part = target.with_name(target.name + '.part')
    with _download_lock:
        existing = find_model(name)
        if existing and existing.stat().st_size >= entry['bytes'] and model_format(existing) not in ('invalid', 'unsupported-gguf'):
            return existing, 'ready'
        target.parent.mkdir(parents=True, exist_ok=True)
        offset = part.stat().st_size if part.exists() else 0
        try:
            headers = {'User-Agent': 'CatSeek-R1/1.x'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(entry['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                # Validate exact Content-Range and total bytes before resuming.
                content_range = response.headers.get('Content-Range', '').strip()
                resume = False
                expected_total = None
                if response.status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if not match:
                        return None, 'Malformed Content-Range; refusing partial download.'
                    start, end, total = map(int, match.groups())
                    if start != offset or end < start or total <= end or (offset == 0 and start != 0):
                        return None, 'Unexpected Content-Range; refusing partial download.'
                    length = response.headers.get('Content-Length')
                    if length is not None and int(length) != end - start + 1:
                        return None, 'Content-Range length mismatch.'
                    resume = offset > 0
                    expected_total = total
                    expected_chunk = end - start + 1
                elif response.status != 200:
                    return None, f'Unexpected HTTP status {response.status}.'
                else:
                    expected_chunk = None
                written = 0
                with part.open('ab' if resume else 'wb') as dest:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if expected_chunk is not None and written > expected_chunk:
                            return None, 'Partial response exceeded advertised range.'
                        dest.write(chunk)
                if expected_chunk is not None and written != expected_chunk:
                    return None, 'Partial response truncated; retry to resume.'
                if expected_total is not None and part.stat().st_size != expected_total:
                    return None, 'Partial checkpoint incomplete; retry to resume.'
            if part.stat().st_size < entry['bytes']:
                return None, 'Checkpoint download incomplete; retry chat to resume.'
            fmt = model_format(part)
            if fmt in ('invalid', 'unsupported-gguf'):
                return None, f'GGUF format check failed for {name}: {fmt}.'
            part.replace(target)
            return target, 'ready'
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            return None, f'Could not download pretrained GGUF: {exc}. Retry when online.'


_build_attempted = False
_build_lock = threading.Lock()


def ensure_pretrained_bitnet(name='bitnet-2b-4t') -> tuple[pathlib.Path | None, pathlib.Path | None, str]:
    """Verify compatibility with a real model load; try all installed runtimes."""
    global _build_attempted
    model, download_status = provision_pretrained_model(name)
    if model is None:
        return None, None, download_status
    fmt = model_format(model)
    if fmt in ('invalid', 'unsupported-gguf'):
        return None, model, f'GGUF format check failed for {name}: {fmt}.'
    binary, errors = select_working_runtime(model)
    if binary:
        return binary, model, 'ready'
    with _build_lock:
        if not _build_attempted:
            _build_attempted = True
            result = build_compatible_bitnet_runtime()
            binary, more_errors = select_working_runtime(model)
            if binary:
                return binary, model, 'ready'
            errors.extend(more_errors)
        else:
            result = 'Automatic runtime build already attempted.'
    return None, model, (result + f'\nRuntime load check failed for {name} ({fmt}): no installed executable loaded this GGUF. '
                         'A genuinely compatible BitNet build is required. '
                         'Use /mode embedded for the untrained demo.\n' +
                         '\n'.join(errors[-3:]))


def clean_cli_output(raw_stdout: str, full_prompt: str) -> str:
    """Strips echoed prompt and trailing reverse-prompt markers from llama-cli stdout."""
    out = raw_stdout.strip()
    if out.startswith(full_prompt):
        out = out[len(full_prompt):].strip()
    elif 'Assistant:' in out:
        out = out.split('Assistant:')[-1].strip()
    # Strip trailing stop sequence if echoed
    for stop_marker in ('\nUser:', 'User:', '<|eot_id|>', '</s>'):
        if stop_marker in out:
            out = out.split(stop_marker)[0].strip()
    return out


# ==============================================================================
# 4. UNIFIED ENGINE (MATH + NATIVE BITNET CLI + BUILT-IN BITNET b1.58)
# ==============================================================================


class LlamaCppChatAPI:
    """Dependency-free client for a LOCAL llama.cpp llama-server chat API.

    Set CATSEEK_LLAMA_URL to a loopback HTTP endpoint (default 127.0.0.1:8080).
    This Python module contains the client, not llama.cpp's C++ inference runtime
    or pretrained weights.
    """
    def __init__(self, url=None):
        from urllib.parse import urlsplit
        self.base = (url or os.getenv('CATSEEK_LLAMA_URL', 'http://127.0.0.1:8080')).rstrip('/')
        parsed = urlsplit(self.base)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1') or parsed.username or parsed.password:
            raise ValueError('For safety, CatSeek only connects to a local HTTP llama.cpp server.')
        self.model = os.getenv('CATSEEK_LLAMA_MODEL', '').strip()

    def chat(self, prompt, history):
        messages = [{'role': 'system', 'content': 'You are CatSeek R1.x, a helpful local AI assistant. Be accurate and concise.'}]
        for user, assistant in history[-12:]:
            messages.extend([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}])
        messages.append({'role': 'user', 'content': prompt})
        body = {'messages': messages, 'stream': False, 'temperature': 0.7, 'max_tokens': 512}
        if self.model:
            body['model'] = self.model
        req = urllib.request.Request(self.base + '/v1/chat/completions',
            data=json.dumps(body).encode('utf-8'),
            headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=180) as response:
            payload = json.load(response)
        answer = payload['choices'][0]['message']['content']
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('llama.cpp returned an empty chat message')
        return answer.strip()

    def ready(self):
        try:
            with urllib.request.urlopen(self.base + '/health', timeout=1.5) as response:
                return response.status == 200
        except (OSError, ValueError):
            return False


class DrMeowMeowChat:
    """Bilingual English/Chinese cat-themed Dr. Meow Meow-style conversational bot.

    Rule-based, not a pretrained language model or medical professional.
    """
    reflections = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours',
                   'am': 'are', 'you': 'I', 'your': 'my', 'yours': 'mine',
                   'are': 'am', 'myself': 'yourself', 'yourself': 'myself'}
    english_rules = [
        (r".*\bi need (.+)", ["What makes you need {0}, meow?", "Would {0} help you, nya?"]),
        (r".*\bi feel (.+)", ["What makes you feel {0}, meow?", "Tell this kitty more about feeling {0}."]),
        (r".*\bi am (.+)", ["How long have you been {0}, nya?", "Why do you say you are {0}, meow?"]),
        (r".*\bi can't (.+)", ["What's making it hard to {0}, meow?", "What could help you {0}, nya?"]),
        (r".*\bi want (.+)", ["Why do you want {0}, meow?", "What would having {0} mean to you, nya?"]),
        (r".*\bmy (.+)", ["Tell me more about your {0}, meow.", "How is your {0} doing, nya?"]),
        (r".*\bbecause (.+)", ["Is that the main reason, meow?", "Are there other reasons, nya?"]),
        (r".*\b(hello|hi|hey)\b.*", ["Hewwo! Dr. Meow Meow is here! 🐱", "Meow meow! What shall we chat about?"]),
        (r".*\b(bye|goodbye)\b.*", ["Bye-bye, nya! 🐾", "See you soon, meow!"]),
        (r".*\b(why|how|what)\b.*", ["What do you think, meow?", "Tell me a little more, nya!"]),
    ]
    chinese_rules = [
        (r".*(?:你好|您好|嗨|哈喽).*", ["喵喵！你好呀！我是喵喵博士！🐱", "你好喵～ 今天想聊什么呀？"]),
        (r".*(?:再见|拜拜).*", ["拜拜喵！下次再聊～🐾", "再见啦，喵～"]),
        (r".*(?:我觉得|我感觉)(.+)", ["为什么会觉得{0}呢，喵？", "可以多说说这种感觉吗，喵？"]),
        (r".*(?:我想要|我想)(.+)", ["为什么想{0}呢，喵？", "如果能{0}，你会有什么感受呢？喵～"]),
        (r".*(?:因为)(.+)", ["这是最主要的原因吗，喵？", "还有别的原因吗，喵？"]),
        (r".*(?:难过|伤心|焦虑|害怕).*", ["喵～ 我在听。想告诉我发生了什么吗？", "抱抱喵。是什么让你有这样的感觉？"]),
        (r".*(?:开心|高兴|快乐).*", ["太好啦喵！是什么让你这么开心？", "喵呜～ 分享一下好消息吧！"]),
    ]
    def __init__(self):
        self.turn = 0
    def reflect(self, phrase):
        return re.sub(r"\b[a-z]+\b", lambda m: self.reflections.get(m.group(0), m.group(0)), phrase.lower())
    def reply(self, prompt):
        clean = re.sub(r"[?!.,。？！]+$", "", prompt.strip())
        self.turn += 1
        chinese = bool(re.search(r"[\u3400-\u9fff]", clean))
        rules = self.chinese_rules if chinese else self.english_rules
        for pattern, replies in rules:
            m = re.fullmatch(pattern, clean if chinese else clean.lower(), re.I)
            if m:
                template = replies[(self.turn - 1) % len(replies)]
                groups = [g if chinese else self.reflect(g) for g in m.groups()]
                return template.format(*groups)
        if chinese:
            return ["喵～ 能多告诉我一点吗？", "这让你想到了什么呢，喵？", "我在认真听哦，喵～", "然后发生了什么呢？🐱"][(self.turn - 1) % 4]
        return ["Meow! Tell me more about that.", "What comes to mind, nya?", "How does that make you feel, meow?", "I'm listening, purr purr! 🐱"][(self.turn - 1) % 4]


class OpenAIChatClient:
    """Standard-library OpenAI-compatible chat client. Never embeds an API key."""
    def __init__(self):
        self.url = os.getenv('CATSEEK_OPENAI_BASE_URL', 'http://127.0.0.1:8080/v1').rstrip('/')
        self.model = os.getenv('CATSEEK_OPENAI_MODEL', 'catseek-r1')
        self.key = os.getenv('CATSEEK_OPENAI_API_KEY', '')
    def chat(self, prompt, history):
        from urllib.parse import urlsplit
        parsed = urlsplit(self.url)
        if parsed.scheme not in ('https', 'http') or (parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1')):
            raise ValueError('Remote API endpoints must use HTTPS')
        messages = [{'role': 'system', 'content': 'You are CatSeek R1, a helpful coding and general assistant.'}]
        for u, a in history[-12:]:
            messages.extend(({'role': 'user', 'content': u}, {'role': 'assistant', 'content': a}))
        messages.append({'role': 'user', 'content': prompt})
        data = json.dumps({'model': self.model, 'messages': messages, 'stream': False, 'max_tokens': 512}).encode()
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        req = urllib.request.Request(self.url + '/chat/completions', data=data, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        content = result['choices'][0]['message']['content']
        if not isinstance(content, str) or not content.strip():
            raise ValueError('API returned no text')
        return content.strip()


class Engine:
    def __init__(self):
        self.history: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        self.micro_bitnet = MicroBitNetLM(dim=32, hidden_dim=64, layers=2)
        self._active_proc: subprocess.Popen | None = None
        self.cancel_event = threading.Event()
        self.auto_setup_attempted = False
        self.llama_api = LlamaCppChatAPI()
        self.openai_api = OpenAIChatClient()
        self.dr_meow_meow = DrMeowMeowChat()
        self.selected_model = os.getenv('CATSEEK_MODEL', 'bitnet-2b-4t').lower()
        if self.selected_model not in MODELS:
            self.selected_model = 'bitnet-2b-4t'
        self.mode = os.getenv('CATSEEK_MODE', 'auto').lower()
        if self.mode not in ('auto', 'bitnet', 'embedded', 'openai', 'dr-meow-meow'):
            self.mode = 'auto' 

    def cancel_active(self):
        """Terminates any running native BitNet subprocess immediately."""
        self.cancel_event.set()
        proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def clear_history(self):
        with self.lock:
            self.history.clear()

    def reply(self, prompt: str, commit_history: bool = True, progress=None) -> tuple[str, str]:
        s = prompt.strip()
        # A new request resets cancellation only at its own start.
        self.cancel_event.clear()
        low = s.lower()

        if low.startswith('/mode'):
            parts = low.split()
            if len(parts) == 1:
                return 'Current mode: ' + self.mode + '. Choose auto, bitnet, embedded, openai, or dr-meow-meow.', 'mode'
            if len(parts) != 2 or parts[1] not in ('auto', 'bitnet', 'embedded', 'openai', 'dr-meow-meow'):
                return 'Usage: /mode auto|bitnet|embedded|openai|dr-meow-meow', 'mode'
            self.mode = parts[1]
            return 'Backend set to ' + self.mode, 'mode'

        if low.startswith('/model '):
            name = low.split(maxsplit=1)[1].strip()
            if name not in MODELS:
                return 'Model name check failed: choose ' + ', '.join(MODELS), 'model'
            self.selected_model = name
            return f'Active model: {name} (installed: {bool(find_model(name))})', 'model'

        if low in ('/clear', '/new'):
            self.clear_history()
            return 'Conversation cleared.', 'local'

        if low in ('/help', 'help'):
            return (
                'Commands:\n'
                '  /help                     — Show this help message\n'
                '  🛠 CatSeek Build           — Create an HTML playground from the composer\n'
                '  /setup                    — Install or verify native BitNet\n'
                '  /status                   — Show backend and model status\n'
                '  /model <name>             — Select registered pretrained model\n'
                '  /mode auto|bitnet|embedded|openai|dr-meow-meow — Select chat backend\n'
                '  /api                      — Show local API server instructions\n'
                '  /bitnet                   — Inspect live ternary {-1, 0, +1} weight stats\n'
                '  /learn <key> :: <reply>   — Demo memory only (not LLM training)\n'
                '  /math <expr>              — Exact fractions, powers, trig, logs, sqrt, factorial\n'
                '  /python                   — Show Python example\n'
                '  /game                     — Show minimal Tkinter game loop\n'
                '  /clear                    — Clear conversation history\n'
                '  Voice toggle              — Speak answers on macOS using say',
                'local',
            )

        if low == '/api':
            return ('Start the local OpenAI-compatible API with:\n'
                    'python3 catseek_r1_1.x.py --serve --port 8765\n'
                    'POST http://127.0.0.1:8765/v1/chat/completions\n'
                    'GET http://127.0.0.1:8765/v1/models\n'
                    'For external OpenAI-compatible APIs set CATSEEK_OPENAI_BASE_URL, '
                    'CATSEEK_OPENAI_MODEL and CATSEEK_OPENAI_API_KEY, then /mode openai.'), 'api-help'

        if low in ('/bitnet', '/weights'):
            stats = self.micro_bitnet.weight_summary()
            fmt = MODELS[self.selected_model]['type']
            if fmt == 'Q1_0_g128':
                stats = {'architecture': 'Bonsai binary-with-group-scale (Q1_0_g128)',
                         'bits_per_weight': 'runtime-specific; not verified',
                         'active_model': self.selected_model,
                         'note': 'Embedded demo statistics are ternary and do not describe Bonsai weights.'}
            else:
                stats['active_model'] = self.selected_model
            return json.dumps(stats, indent=2), 'bitnet-b1.58-inspect'

        if low.startswith('/learn '):
            body = s[7:].strip()
            if '::' not in body:
                return 'Usage: /learn <keywords> :: <response text>', 'local'
            k, v = body.split('::', 1)
            self.micro_bitnet.custom_memories.append((k.strip(), v.strip()))
            return f'Learned custom rule for "{k.strip()}".', 'bitnet-b1.58-learn'

        if low in ('/setup', '/setpu'):
            return setup_bitnet(self.selected_model, progress), 'setup'

        if low in ('/status', '/model'):
            summary = self.micro_bitnet.weight_summary()
            installed = registered_models()
            runtime = detect()[0]
            selected = installed.get(self.selected_model)
            gguf = model_format(selected)
            valid = bool(selected and selected.stat().st_size >= MODELS[self.selected_model]['bytes'] and gguf == MODELS[self.selected_model]['type'])
            verified = bool(valid and runtime and select_working_runtime(selected)[0])
            return (json.dumps({'app': NAME, 'mode': 'pretrained BitNet auto-provisioning',
                'files': 'OFF (no user-provided files; downloaded pretrained weights are cached)', 'setup_required': not verified, 'first_launch_download_required': not valid,
                'pretrained_llm_bundled': False, 'ready': verified,
                'selected_mode': self.mode, 'native_bitnet_detected': bool(runtime),
                'active_model': self.selected_model,
                'installed_models': {k: str(v) for k, v in installed.items()},
                'runtime_verified': verified,
                'gguf_format': gguf,
                'architecture': ('Bonsai binary-with-group-scale (Q1_0_g128)' if self.selected_model == 'bonsai-1.7b' else summary['architecture']),
                'ternary_weights': summary['total_ternary_weights'],
                'note': 'Auto/bitnet uses real pretrained weights, downloading on first run. Embedded mode remains an untrained demo.'}, indent=2), 'status')

        expression = re.sub(r'^(?:/math|calculate|compute|what is|what\'s)\s+', '', s, flags=re.I).strip(' ?')
        math_requested = low.startswith(('/math ', 'calculate ', 'compute '))
        math_chars = bool(re.fullmatch(r'[0-9.eE\s+*/%()\-^×÷]+', expression)) and any(x.isdigit() for x in expression)
        if math_requested or math_chars:
            try:
                return str(calculate(expression)), 'exact-math'
            except Exception as e:
                return 'Math error: ' + str(e), 'math-error'

        if low == '/python':
            return (
                'Python example:\n\n'
                'def greet(name: str) -> str:\n'
                '    return f"Hello, {name}!"\n\n'
                'print(greet("CatSeek"))',
                'local-template',
            )

        if low == '/game':
            return (
                'Minimal tkinter game loop:\n\n'
                'import tkinter as tk\n'
                'r = tk.Tk()\n'
                'c = tk.Canvas(r, width=400, height=300, bg="black")\n'
                'c.pack()\n'
                'x = 0\n'
                'def tick():\n'
                '    global x\n'
                '    c.delete("all")\n'
                '    c.create_oval(x, 120, x + 30, 150, fill="cyan")\n'
                '    x = (x + 3) % 400\n'
                '    r.after(16, tick)\n'
                'tick()\n'
                'r.mainloop()',
                'local-template',
            )

        with self.lock:
            hist_snapshot = list(self.history[-12:])

        route = 'embedded-bitnet-experimental'
        if self.mode == 'dr-meow-meow':
            out, route = self.dr_meow_meow.reply(s), 'dr-meow-meow-rule-based'
        elif self.mode == 'openai':
            try:
                out, route = self.openai_api.chat(s, hist_snapshot), 'openai-compatible-api'
            except Exception as exc:
                return f'OpenAI-compatible API error: {exc}', 'api-error'
        else:
            if self.mode in ('auto', 'bitnet'):
                try:
                    binary, model, provision = ensure_pretrained_bitnet(self.selected_model)
                except (OSError, subprocess.SubprocessError) as exc:
                    return f'Pretrained BitNet provisioning failed: {exc}', 'model-required'
            else:
                binary, model, provision = None, None, 'embedded'
            if binary and (model or provision == 'huggingface-auto'):
                # Native inference requires an installed compatible executable and pretrained weights.
                full_prompt = 'You are CatSeek R1, a helpful assistant.\n'
                for u, a in hist_snapshot[-6:]:
                    full_prompt += f'User: {u}\nAssistant: {a}\n'
                full_prompt += f'User: {s}\nAssistant:'
                try:
                    model_args = ['-m', str(model)] if model else ['-hf', BITNET_HF_REPO]
                    proc = subprocess.Popen(native_command(binary, model, full_prompt, 256),
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, errors='replace')
                    self._active_proc = proc
                    if self.cancel_event.is_set():
                        proc.terminate()
                    try:
                        # Hugging Face -hf may download the checkpoint on a cold cache.
                        # Local -m inference retains the shorter timeout.
                        native_timeout = 3600 if model is None else 180
                        stdout, stderr = proc.communicate(timeout=native_timeout)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.communicate()
                        return f'Native BitNet timed out after {native_timeout}s.', 'native-error'
                    finally:
                        self._active_proc = None
                    if self.cancel_event.is_set():
                        return 'Generation cancelled.', 'cancelled'
                    if native_failure(stdout, stderr, proc.returncode):
                        diagnostic = native_diagnostic(binary, stdout, stderr, proc.returncode)
                        if any(term in stderr.lower() for term in ('not a multiple of block size', 'iq4_nl_4_4', 'i2_s', 'unsupported tensor type')):
                            return (f'Native runtime tensor load check failed for {self.selected_model} '
                                    f'({model_format(model)}): {stderr[-1200:]}'), 'model-required'
                        return f'Native runtime execution check failed for {self.selected_model}: ' + diagnostic, 'native-error'
                    out = clean_cli_output(stdout, full_prompt)
                    if not out:
                        return 'Native BitNet returned an empty answer.', 'native-error'
                    route = 'native-bitnet-gguf'
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    return f'Native BitNet launch error: {exc}', 'native-error'
            elif self.mode in ('auto', 'bitnet'):
                return ('Real pretrained BitNet unavailable: ' + provision + '\nNo scripted neural fallback was used.'), 'model-required'
            else:
                try:
                    out = self.micro_bitnet.generate(s, hist_snapshot)
                except (ValueError, OverflowError, ArithmeticError) as exc:
                    return f'Embedded BitNet numerical error: {exc}', 'bitnet-error'
        if self.cancel_event.is_set():
            return 'Generation cancelled.', 'cancelled'
        if commit_history and not self.cancel_event.is_set():
            with self.lock:
                if self.cancel_event.is_set():
                    return 'Generation cancelled.', 'cancelled'
                self.history.append((s, out))
                self.history = self.history[-24:]
        return out, route



# ==============================================================================
# CATSEEK BUILD — standard-library starter project scaffolder
# ==============================================================================

def catseek_build(description: str) -> pathlib.Path:
    """Create a generic, model-free HTML/CSS/JS playground for any request."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError('Enter a project description.')
    if len(description) > 100_000:
        raise ValueError('Project description is too long (100,000 characters max).')

    import html
    from datetime import datetime

    slug = re.sub(r'[^a-z0-9]+', '-', description.lower()).strip('-')[:48].strip('-') or 'html-project'
    # Store generated projects beside the running .py, not in a model directory.
    root = pathlib.Path(__file__).resolve().parent
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    project = root / f'{slug}-{stamp}'
    project.mkdir(mode=0o700)

    escaped = html.escape(description, quote=True)
    # The user's request is displayed as text, never treated as markup or code.
    page = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__REQUEST__ — CatSeek Build</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}body{margin:0;background:#0b1020;color:#e5edff}
header{padding:24px 5vw;background:#131e34;border-bottom:1px solid #30425f}
h1{margin:0 0 8px;color:#8fc7ff;font-size:clamp(1.4rem,3vw,2rem)}
main{max-width:1100px;margin:auto;padding:24px;display:grid;gap:18px}
.panel{padding:20px;background:#152139;border:1px solid #30425f;border-radius:12px}
.request{white-space:pre-wrap;overflow-wrap:anywhere}
textarea{width:100%;height:220px;resize:vertical;background:#080e1d;color:#d9f2ff;
 border:1px solid #47638a;border-radius:8px;padding:14px;font:14px/1.5 ui-monospace,monospace}
button{background:#0b0b0b;color:#7ebaff;border:1px solid #44618c;border-radius:8px;
 padding:10px 18px;cursor:pointer;font-weight:700}
button:hover{border-color:#9acbff}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
iframe{width:100%;height:420px;background:white;border:1px solid #47638a;border-radius:8px}
canvas{display:block;max-width:100%;background:#0c1528;border:1px solid #405878;border-radius:8px}
small{color:#b7c9e3}
</style>
</head>
<body>
<header><h1>CatSeek Build · HTML Playground</h1><small>Standalone project · no model required</small></header>
<main>
<section class="panel"><h2>Your request</h2><div class="request">__REQUEST__</div></section>
<section class="panel"><h2>Canvas sandbox</h2>
<canvas id="canvas" width="640" height="120" aria-label="Canvas starter"></canvas>
<small>Use the editor below to create your own HTML, CSS, and JavaScript.</small></section>
<section class="panel"><h2>HTML / CSS / JavaScript editor</h2>
<label for="playground">Paste a complete HTML document or an HTML/CSS/JS snippet:</label>
<textarea id="playground" spellcheck="false"><h1>Hello from CatSeek!</h1>
<p>Replace this snippet with your own project code.</p>
<canvas id="game" width="480" height="180" style="background:#111827"></canvas>
<script>
const c = document.getElementById('game');
const ctx = c.getContext('2d');
let x = 0;
function draw() {
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = '#60a5fa'; ctx.fillRect(x, 75, 30, 30);
  x = (x + 2) % c.width;
  requestAnimationFrame(draw);
}
draw();
</script></textarea>
<div class="controls"><button id="render" type="button">Run preview</button>
<button id="clear" type="button">Clear editor</button>
<small id="status" role="status">Ready</small></div></section>
<section class="panel"><h2>Live preview</h2>
<iframe id="preview" title="HTML playground preview" sandbox="allow-scripts"></iframe>
<small>Preview scripts run in an isolated iframe. Do not paste untrusted code.</small></section>
</main>
<script>
'use strict';
const editor = document.getElementById('playground');
const preview = document.getElementById('preview');
const status = document.getElementById('status');
function render() {
  // srcdoc runs inside a sandboxed, opaque-origin iframe; no eval is used.
  preview.srcdoc = editor.value;
  status.textContent = 'Preview updated';
}
document.getElementById('render').addEventListener('click', render);
document.getElementById('clear').addEventListener('click', () => {
  editor.value = ''; render(); editor.focus();
});
let pending;
editor.addEventListener('input', () => {
  clearTimeout(pending);
  pending = setTimeout(render, 250);
});
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#0c1528'; ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.fillStyle = '#8fc7ff'; ctx.font = '24px system-ui';
ctx.fillText('Your canvas is ready', 22, 65);
render();
</script>
</body>
</html>
'''.replace('__REQUEST__', escaped)
    (project / 'index.html').write_text(page, encoding='utf-8')
    (project / 'README.md').write_text(
        '# CatSeek Build — HTML Playground\n\n'
        '## Your request\n\n' + description + '\n\n'
        '## Run\n\nOpen `index.html` in a modern web browser. '
        'Edit or paste HTML/CSS/JS into the playground and select Run preview.\n\n'
        'This generic sandbox does not automatically implement the requested game or app; '
        'it provides an editable live-preview starter for every prompt. '
        'Preview code is isolated in a sandboxed iframe.\n', encoding='utf-8')
    return project


def is_build_request(prompt: str) -> bool:
    """Route explicit project requests locally; ordinary conversation stays in chat."""
    stripped = prompt.strip()
    if stripped.lower().startswith('/build '):
        return True
    return bool(re.match(r'^(?:write|make|create|build|generate|code|develop)\s+.+\b(?:in\s+html|in\s+javascript|in\s+js|web\s*page|website|html|javascript|landing page)\b', stripped, re.I))


def build_reply(prompt: str) -> tuple[str, str]:
    description = re.sub(r'^/build\s+', '', prompt, flags=re.I)
    project = catseek_build(description)
    return (f'Created HTML project: {project}\n'
            'Files: index.html, README.md\n'
            f'Open in browser: {project / "index.html"}\n'
            'This is a live HTML playground starter, not an automatically implemented game.',
            'catseek-build')


# ==============================================================================
# 5. TKINTER GUI (FIXED CONCURRENCY & GENERATION LOCKING)
# ==============================================================================

class GUI:
    def __init__(self, root):
        self.root = root
        self.engine = Engine()
        self.q = queue.Queue()
        self.busy = False
        self.generation = 0
        self.voice_enabled = False
        self.voice_proc = None

        root.title(NAME)
        root.geometry('980x680')
        root.minsize(720, 480)
        root.configure(bg='#080d17')

        head = tk.Frame(root, bg='#101c30', padx=18, pady=16)
        head.pack(fill='x')
        tk.Label(
            head,
            text='🐱  CATSEEK R1 1.x',
            bg='#101c30',
            fg='#8fc7ff',
            font=('Helvetica', 20, 'bold'),
        ).pack(side='left')
        self.status = tk.Label(head, text='Pretrained BitNet · auto', bg='#101c30', fg='#a9bed8')
        self.status.pack(side='right')

        body = tk.Frame(root, bg='#080d17')
        body.pack(fill='both', expand=True)

        side = tk.Frame(body, bg='#101c30', width=190, padx=12, pady=16)
        side.pack(side='left', fill='y')
        side.pack_propagate(False)

        buttons = [
            ('New chat', self.new),
            ('🛠 CatSeek Build', self.build_project),
            ('Stop generation', self.stop),
            ('Voice: OFF', self.toggle_voice),
            ('Setup BitNet', lambda: self.submit('/setup')),
            ('Model status', lambda: self.submit('/status')),
            ('BitNet mode', lambda: self.submit('/mode bitnet')),
            ('Embedded mode', lambda: self.submit('/mode embedded')),
            ('OpenAI API mode', lambda: self.submit('/mode openai')),
            ('Dr. Meow Meow mode', lambda: self.submit('/mode dr-meow-meow')),
            ('Auto mode', lambda: self.submit('/mode auto')),
            ('BitNet weights', lambda: self.submit('/bitnet')),
            ('Math demo', lambda: self.submit('/math (12+8)*3')),
            ('What is BitNet?', lambda: self.submit('What is BitNet 1.58?')),
            ('Python demo', lambda: self.submit('/python')),
            ('Game demo', lambda: self.submit('/game')),
        ]
        for label, cmd in buttons:
            btn = tk.Button(
                side,
                text=label,
                command=cmd,
                bg='black',
                fg='blue',
                activebackground='black',
                activeforeground='blue',
                relief='flat',
                pady=8,
            )
            btn.pack(fill='x', pady=4)
            if label == 'Voice: OFF':
                self.voice_button = btn

        main = tk.Frame(body, bg='#080d17')
        main.pack(side='left', fill='both', expand=True)

        self.chat = scrolledtext.ScrolledText(
            main,
            bg='#080d17',
            fg='#e0edff',
            insertbackground='white',
            wrap='word',
            font=('Menlo', 11),
            relief='flat',
            padx=18,
            pady=16,
            state='disabled',
        )
        self.chat.pack(fill='both', expand=True)

        bar = tk.Frame(main, bg='#101c30', padx=12, pady=12)
        bar.pack(fill='x')

        self.input = tk.Text(
            bar,
            height=3,
            bg='#17283f',
            fg='#f4f8ff',
            insertbackground='white',
            font=('Helvetica', 12),
            relief='flat',
            wrap='word',
            padx=10,
            pady=8,
        )
        self.input.pack(side='left', fill='x', expand=True)
        self.input.bind('<Return>', self.enter)

        self.send = tk.Button(
            bar,
            text='SEND ↑',
            command=self.submit,
            bg='black',
            fg='blue',
            activebackground='black',
            activeforeground='blue',
            relief='flat',
            padx=14,
            pady=12,
        )
        self.send.pack(side='right', padx=(10, 0))

        self.append(
            f'{NAME} · Pretrained BitNet automatic first-run download · FILES OFF (app assets)\n'
            'Type /help for commands, /bitnet to inspect ternary {-1, 0, +1} weights, or ask a question.\n\n'
        )
        self.refresh()
        self.root.after(50, self.poll)
        self.input.focus_set()
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def close(self):
        self.stop()
        self.root.destroy()

    def append(self, s: str):
        self.chat.configure(state='normal')
        self.chat.insert('end', s)
        self.chat.configure(state='disabled')
        self.chat.see('end')

    def refresh(self):
        self.status.configure(text=f'{self.engine.mode.upper()} · FILES OFF')

    def enter(self, event):
        if event.state & 0x1:  # Shift+Enter inserts newline
            return None
        self.submit()
        return 'break'

    def toggle_voice(self):
        """Optional speech synthesis on macOS, with no extra Python packages."""
        self.voice_enabled = not self.voice_enabled
        self.voice_button.configure(text='Voice: ON' if self.voice_enabled else 'Voice: OFF')
        if not self.voice_enabled:
            self.stop_voice()
        elif not shutil.which('say'):
            self.append('Voice output requires macOS `say` (text chat still works).\n\n')
            self.voice_enabled = False
            self.voice_button.configure(text='Voice: OFF')

    def stop_voice(self):
        proc = self.voice_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        self.voice_proc = None

    def speak(self, answer):
        if not self.voice_enabled or not shutil.which('say'):
            return
        self.stop_voice()
        # Passing speech as an argument, never as shell source.
        speech = re.sub(r'```[\s\S]*?```', 'Code example omitted.', answer)
        speech = re.sub(r'[`*_#]', '', speech).strip()[:3500]
        if speech:
            try:
                self.voice_proc = subprocess.Popen(['say', speech], stdout=subprocess.DEVNULL,
                                                    stderr=subprocess.DEVNULL)
            except OSError:
                self.voice_proc = None

    def stop(self):
        was_busy = self.busy
        self.engine.cancel_active()
        self.stop_voice()
        self.generation += 1
        self.busy = False
        self.send.configure(state='normal')
        if was_busy:
            self.append('Generation stopped.\n\n')

    def new(self):
        # Increment generation & cancel any active subprocess so stale workers can't clobber UI state
        self.generation += 1
        self.engine.cancel_active()
        self.stop_voice()
        self.engine.clear_history()
        self.busy = False
        self.send.configure(state='normal')
        self.chat.configure(state='normal')
        self.chat.delete('1.0', 'end')
        self.chat.configure(state='disabled')
        self.append('New CatSeek chat\n\n')
        self.refresh()

    def build_project(self):
        """Build from the composer without invoking a model or executing output."""
        if self.busy:
            self.append('Wait for the current response before building.\n\n')
            return
        description = self.input.get('1.0', 'end-1c').strip()
        if not description:
            self.append('CATSEEK BUILD\nType a project description first (for example: write pong in html).\n\n')
            self.input.focus_set()
            return
        try:
            project = catseek_build(description)
        except (OSError, ValueError) as exc:
            self.append(f'CATSEEK BUILD\nCould not create project: {exc}\n\n')
            return
        self.input.delete('1.0', 'end')
        self.append(f'YOU\n{description}\n\nCATSEEK BUILD\n'
                    f'Created starter project: {project}\n'
                    'Files: index.html, README.md\n'
                    f'Open in browser: {project / "index.html"}\n'
                    'Nothing was executed automatically.\n\n[catseek-build]\n\n')

    def submit(self, forced=None):
        if self.busy:
            return
        prompt = forced if isinstance(forced, str) else self.input.get('1.0', 'end-1c').strip()
        if not prompt:
            return
        if not isinstance(forced, str):
            self.input.delete('1.0', 'end')

        self.append(f'YOU\n{prompt}\n\nCATSEEK\n')
        if is_build_request(prompt):
            try:
                answer, route = build_reply(prompt)
                self.append(f'{answer}\n\n[{route}]\n\n')
            except (OSError, ValueError) as exc:
                self.append(f'Build failed: {exc}\n\n[build-error]\n\n')
            return
        self.busy = True
        self.send.configure(state='disabled')
        gen = self.generation

        def work():
            try:
                ans, route = self.engine.reply(prompt, commit_history=False,
                    progress=lambda msg: self.q.put((gen, prompt, msg, 'setup-progress')))
                self.q.put((gen, prompt, ans, route))
            except Exception as e:
                self.q.put((gen, prompt, f'Error: {e}', 'error'))

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                try:
                    gen, original_prompt, answer, route = self.q.get_nowait()
                except queue.Empty:
                    break
                try:
                    if gen != self.generation:
                        continue
                    if route == 'setup-progress':
                        self.append(answer + '\n')
                        continue
                    if route not in ('cancelled', 'error', 'native-error', 'model-required', 'api-error', 'bitnet-error', 'math-error') and not answer.startswith('Error:'):
                        if not original_prompt.startswith('/'):
                            with self.engine.lock:
                                if self.engine.cancel_event.is_set() or gen != self.generation:
                                    continue
                                self.engine.history.append((original_prompt, answer))
                                self.engine.history = self.engine.history[-24:]
                    self.append(f'{answer}\n\n[{route}]\n\n')
                    if route not in ('error', 'native-error', 'llm-error', 'model-required', 'cancelled'):
                        self.speak(answer)
                except Exception as exc:
                    try:
                        self.append(f'GUI response error: {exc}\n\n')
                    except Exception:
                        pass
                finally:
                    if gen == self.generation and route != 'setup-progress':
                        self.busy = False
                        self.send.configure(state='normal')
                        self.refresh()
        finally:
            self.root.after(50, self.poll)


# ==============================================================================
# 6. CLI & SELF-TEST ENTRYPOINT
# ==============================================================================


class CatSeekAPIHandler(http.server.BaseHTTPRequestHandler):
    """Local-only OpenAI-compatible JSON API; non-streaming chat completions."""
    engine = None
    def log_message(self, fmt, *args):
        pass
    def send_json(self, code, payload):
        data = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path in ('/v1/models', '/models'):
            return self.send_json(200, {'object':'list', 'data':[
                {'id': 'catseek-r1', 'object':'model', 'owned_by':'catseek'},
                {'id': 'catseek-dr-meow-meow', 'object':'model', 'owned_by':'catseek'},
                {'id': 'catseek-embedded', 'object':'model', 'owned_by':'catseek'}]})
        if self.path == '/health':
            return self.send_json(200, {'status':'ok'})
        self.send_json(404, {'error': {'message':'Not found'}})
    def do_POST(self):
        if self.path not in ('/v1/chat/completions', '/chat/completions'):
            return self.send_json(404, {'error': {'message':'Not found'}})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1024 * 1024:
                raise ValueError('Request body must be 1 byte to 1 MiB')
            body = json.loads(self.rfile.read(size))
            if body.get('stream'):
                return self.send_json(400, {'error': {'message':'Streaming is not implemented'}})
            messages = body.get('messages', [])
            if not isinstance(messages, list) or not messages:
                raise ValueError('messages must be a nonempty list')
            prompt = next((m.get('content') for m in reversed(messages)
                           if isinstance(m, dict) and m.get('role') == 'user'), None)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('A nonempty user text message is required')
            model = body.get('model', 'catseek-r1')
            if model not in ('catseek-r1', 'catseek-dr-meow-meow', 'catseek-embedded'):
                raise ValueError('Unknown model')
            eng = Engine()
            eng.mode = {'catseek-dr-meow-meow':'dr-meow-meow', 'catseek-embedded':'embedded',
                        'catseek-r1':'auto'}[model]
            history = []
            pending = None
            for m in messages[:-1]:
                if not isinstance(m, dict) or not isinstance(m.get('content'), str):
                    continue
                if m.get('role') == 'user':
                    pending = m['content']
                elif m.get('role') == 'assistant' and pending is not None:
                    history.append((pending, m['content']))
                    pending = None
            eng.history = history[-12:]
            answer, route = eng.reply(prompt, commit_history=False)
            if route in ('native-error','api-error','model-required','bitnet-error'):
                return self.send_json(503, {'error': {'message': answer, 'type':'backend_error'}})
            self.send_json(200, {'id':'chatcmpl-'+uuid.uuid4().hex, 'object':'chat.completion',
                'created':int(time.time()), 'model':model, 'choices':[{'index':0,
                'message':{'role':'assistant','content':answer}, 'finish_reason':'stop'}],
                'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0},
                'catseek_backend':route})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {'error': {'message':str(exc), 'type':'invalid_request_error'}})
        except Exception as exc:
            self.send_json(500, {'error': {'message':str(exc), 'type':'server_error'}})


def serve_api(port):
    server = http.server.ThreadingHTTPServer(('127.0.0.1', port), CatSeekAPIHandler)
    print(f'CatSeek API: http://127.0.0.1:{port}/v1 (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

def main():
    if '--serve' in sys.argv:
        port = 8765
        if '--port' in sys.argv:
            idx = sys.argv.index('--port')
            try:
                port = int(sys.argv[idx + 1])
                if not 1 <= port <= 65535:
                    raise ValueError()
            except (IndexError, ValueError):
                print('Error: --port requires a numeric value from 1 to 65535.', file=sys.stderr)
                sys.exit(2)
        serve_api(port)
        return
    if '--self-test' in sys.argv:
        assert calculate('2+3*4') == 14
        assert calculate('(12+8)*3') == 60
        assert calculate('1/3+1/6').startswith('1/2')
        assert calculate('sqrt(81)') == '9'
        assert calculate('factorial(6)') == 720
        assert abs(float(calculate('sin(pi/2)')) - 1) < 1e-12
        eng = Engine()
        status_out, status_route = eng.reply('/status')
        assert 'pretrained_llm_bundled' in status_out and status_route == 'status'
        bitnet_out, bitnet_route = eng.reply('/bitnet')
        stats = json.loads(bitnet_out)
        assert stats['total_ternary_weights'] > 0 and bitnet_route == 'bitnet-b1.58-inspect'
        eng.mode = 'embedded'
        chat_out, chat_route = eng.reply('What is BitNet 1.58?')
        assert chat_route in ('embedded-bitnet-experimental', 'native-bitnet-gguf') and bool(chat_out)
        eng.reply('/mode dr-meow-meow')
        assert eng.reply('I feel happy')[1] == 'dr-meow-meow-rule-based'
        eng.reply('/mode embedded')
        print('PASS: math, status, BitNet inspection and embedded neural chat routing.')
        return

    if '--prompt' in sys.argv:
        i = sys.argv.index('--prompt')
        # CLI prompts must never trigger automatic native provisioning by default.
        # --mode may appear before or after --prompt.
        args = sys.argv[1:]
        allowed_modes = ('auto', 'bitnet', 'embedded', 'openai', 'dr-meow-meow')
        cli_mode = 'embedded'
        if '--mode' in args:
            mode_idx = args.index('--mode')
            if mode_idx + 1 >= len(args) or args[mode_idx + 1] not in allowed_modes:
                print('Error: --mode must be one of: ' + ', '.join(allowed_modes), file=sys.stderr)
                sys.exit(2)
            cli_mode = args[mode_idx + 1]
            del args[mode_idx:mode_idx + 2]
        prompt_idx = args.index('--prompt')
        prompt_text = ' '.join(args[prompt_idx + 1:]).strip()
        if not prompt_text:
            print('Error: --prompt requires a non-empty prompt string.', file=sys.stderr)
            sys.exit(1)
        if is_build_request(prompt_text):
            ans, route = build_reply(prompt_text)
        else:
            engine = Engine()
            engine.mode = cli_mode
            ans, route = engine.reply(prompt_text, progress=lambda msg: print(msg, flush=True))
        print(f'{ans}\n[{route}]')
        return

    if not TK_AVAILABLE:
        print('Tkinter is not available in this Python environment. Use --prompt "..." or --self-test.', file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()