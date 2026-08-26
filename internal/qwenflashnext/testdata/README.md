# Pinned upstream fixtures

Source: `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`

Downloaded from Hugging Face `resolve/<revision>/...` on 2026-08-26.

| Fixture / upstream file | SHA-256 |
|---|---|
| `chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| upstream `tokenizer_config.json` | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| upstream `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |

`special_tokens.json` is the deterministic, reviewable extraction of the tokens relevant to chat, thinking, tools, padding, and stopping. The upstream 12.8 MB tokenizer is identified by hash rather than duplicated.
