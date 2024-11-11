### `utils.py` - Tokenizer Setup with `spm_tokenizer`

#### Function `spm_tokenizer`
```python
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model

def spm_tokenizer(metadata):
    tokens = metadata["tokenizer.ggml.tokens"]
    bos = metadata["tokenizer.ggml.bos_token_id"].item()
    eos = metadata["tokenizer.ggml.eos_token_id"].item()
    unk = metadata["tokenizer.ggml.unknown_token_id"].item()
```
- **Purpose**: Initializes a SentencePiece tokenizer based on metadata, such as tokens, BOS (beginning of sentence), EOS (end of sentence), and unknown tokens.
  
#### Creating Normalizer and Trainer Specs
```python
    normalizer_spec = model.NormalizerSpec(
        name="identity",
        precompiled_charsmap=b"",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        normalization_rule_tsv=b"",
    )
    trainer_spec = model.TrainerSpec(
        model_type="BPE",
        vocab_size=len(tokens),
        input_format="text",
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        vocabulary_output_piece_score=True,
        byte_fallback=True,
        unk_id=unk,
        bos_id=bos,
        eos_id=eos,
        pad_id=-1,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        pretokenization_delimiter="",
    )
```
- **Purpose**: Defines how the tokenizer should handle normalization and training. Sets token IDs for BOS, EOS, unknown, etc., and configures byte fallback, vocabulary scores, and other rules for BPE (Byte Pair Encoding).

#### Adding Tokens and Compiling Tokenizer
```python
    m = model.ModelProto(trainer_spec=trainer_spec, normalizer_spec=normalizer_spec)
    scores = metadata.get("tokenizer.ggml.scores", None)
    scores = scores.tolist() if scores is not None else None
    token_types = metadata.get("tokenizer.ggml.token_type", None)
    token_types = token_types.tolist() if token_types is not None else None

    for i, token in enumerate(tokens):
        score = scores[i] if scores else 0
        token_type = token_types[i] if token_types else 0
        m.pieces.append(
            model.ModelProto.SentencePiece(piece=token, score=score, type=token_type)
        )
    tokenizer = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
    return tokenizer
```
- **Purpose**: Iterates through tokens, adding each to the tokenizer with its score and type. This setup allows the tokenizer to handle a custom vocabulary defined by the metadata.

---

### `generate.py` - Main Generation Script

#### Importing Required Modules
```python
import argparse
import time

import mlx.core as mx
import models
```
- **Purpose**: Imports necessary libraries (`argparse` for command-line parsing, `time` for performance tracking) and modules (`mlx` core and `models` for MLX-related functionality).

#### `generate` Function - Model Text Generation
```python
def generate(
    model: models.Model,
    tokenizer: models.GGUFTokenizer,
    prompt: str,
    max_tokens: int,
    temp: float = 0.0,
):
    prompt = tokenizer.encode(prompt)

    tic = time.time()
    tokens = []
    skip = 0
```
- **Purpose**: Converts the input prompt to tokens using the tokenizer and prepares for token generation.
- **Parameters**:
  - `model`: The loaded language model.
  - `tokenizer`: The `GGUFTokenizer` for encoding and decoding tokens.
  - `prompt`: The initial text input.
  - `max_tokens`: Maximum tokens for generation.
  - `temp`: Sampling temperature (controls randomness).

#### Generating and Printing Tokens
```python
    for token, n in zip(
        models.generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        if n == 0:
            prompt_time = time.time() - tic
            tic = time.time()

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)
    print(tokenizer.decode(tokens)[skip:], flush=True)
    gen_time = time.time() - tic
```
- **Purpose**:
  - Loops through generated tokens, breaking if an EOS token is encountered.
  - Tracks time for prompt encoding and token generation.
  - Decodes and prints generated tokens progressively.
  - Maintains a `skip` variable to print new text without reprinting previous tokens.

#### Final Reporting of Generation Speed
```python
    print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return
    prompt_tps = prompt.size / prompt_time
    gen_tps = (len(tokens) - 1) / gen_time
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")
```
- **Purpose**: Calculates and displays tokens-per-second for prompt processing and token generation. Helps evaluate model efficiency.

---

#### `__main__` Block - Parsing Arguments and Running Generation
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--gguf",
        type=str,
        help="The GGUF file to load (and optionally download).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="The Hugging Face repo if downloading from the Hub.",
    )

    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()
    mx.random.seed(args.seed)
    model, tokenizer = models.load(args.gguf, args.repo)
    generate(model, tokenizer, args.prompt, args.max_tokens, args.temp)
```
- **Purpose**:
  - Defines command-line arguments for specifying GGUF file path, Hugging Face repository, prompt, max tokens, temperature, and seed.
  - Loads the model and tokenizer using `models.load`, then initiates generation with the `generate` function.
