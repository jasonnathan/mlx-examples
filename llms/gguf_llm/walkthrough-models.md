### Importing Modules and Setting Up `ModelArgs`
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import utils
from huggingface_hub import snapshot_download
```
- **Purpose**: Imports required modules:
  - `dataclasses` for defining data structure.
  - `Path` for file handling.
  - MLX core, NN, and utilities.
  - `snapshot_download` from `huggingface_hub` to download model snapshots.

```python
@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    context_length: int
    num_key_value_heads: int = None
    rope_theta: float = 10000
    rope_traditional: bool = False
    model_type: str = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
```
- **Purpose**: `ModelArgs` holds configuration parameters for the model, such as hidden layer sizes, vocab size, context length, etc.

---

### Initializing `ModelArgs`
```python
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"factor", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if self.rope_scaling["type"] != "linear":
                raise ValueError("rope_scaling 'type' currently only supports 'linear'")
```
- **Purpose**: Sets default values if parameters are not provided, performs validation checks for rope scaling.

---

### `Attention` Class Definition
```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.repeats = n_heads // n_kv_heads
```
- **Purpose**: Initializes an attention mechanism.
- **Parameters**:
  - Uses `hidden_size` for input dimensions, `num_attention_heads`, `num_key_value_heads`.

---

### Attention Projections and RoPE
```python
        head_dim = args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        rope_scale = (
            1 / args.rope_scaling["factor"]
            if args.rope_scaling is not None and args.rope_scaling["type"] == "linear"
            else 1
        )
        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
            scale=rope_scale,
        )
```
- **Purpose**: Sets up attention projections and scaling.
- **RoPE**: Rotational positional embeddings (RoPE) applied based on `head_dim`, with `rope_theta` and scaling options.

---

### Attention Forward Method
```python
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
```
- **Purpose**: Computes attention scores by projecting inputs into queries, keys, and values.
  
---

### Applying Attention with Masking
```python
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)
```
- **Purpose**: Adds RoPE-based rotational embeddings and caches keys and values across layers.
  
---

### MLP Layer Class
```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
```
- **Purpose**: Implements a multi-layer perceptron (MLP) with gated activation and up/down projections.

---

### Transformer Block Definition
```python
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
```
- **Purpose**: Builds a single transformer block with attention, MLP, and RMS normalization.

---

### Forward Method for Transformer Block
```python
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache
```
- **Purpose**: Processes input through self-attention, MLP, and layer normalization.

---

### LlamaModel Class Initialization
```python
class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
```
- **Purpose**: Defines the main model architecture with token embedding, transformer blocks, and final normalization.

---

### Forward Method in LlamaModel
```python
    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)
        if cache is None:
            cache = [None] * len(self.layers)
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, mask, cache[e])
        return self.norm(h), cache
```
- **Purpose**: Applies embeddings, attention mask, and transformer blocks sequentially.

---

### Model Wrapper
```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model = LlamaModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out, cache = self.model(inputs, cache)
        return self.lm_head(out), cache
```
- **Purpose**: Wraps LlamaModel with final linear projection to vocabulary logits.

---

### GGUF Tokenizer Class
```python
class GGUFTokenizer:
    def __init__(self, metadata):
        self._tokenizer = utils.spm_tokenizer(metadata)

    def encode(self, s: str) -> mx.array:
        return mx.array([self._tokenizer.bos_id()] + self._tokenizer.encode(s))

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_id

()

    def decode(self, toks: List[int]) -> str:
        return self._tokenizer.decode(toks)
```
- **Purpose**: Handles encoding/decoding of tokens with Start and End token IDs.

---

### Generating Model Output
```python
def generate(prompt: mx.array, model: Model, temp: float = 0.0):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y
```
- **Purpose**: Generates text from model predictions based on sampling (with/without temperature).
