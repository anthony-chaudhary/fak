# Example Output

Command:

```bash
go run ./cmd/dropindemo -selfcheck
```

Representative output:

```text
== dropindemo -selfcheck: resolve each entry point through internal/dropin (browserless) ==
example gateway: http://127.0.0.1:8080

  claude         PASS   fak guard -- claude    -> anthropic · injects ANTHROPIC_BASE_URL
  codex          PASS   fak guard -- codex     -> openai · injects OPENAI_BASE_URL
  opencode       PASS   fak guard -- opencode  -> openai · injects OPENAI_BASE_URL
  aider          PASS   fak guard -- aider     -> openai · injects OPENAI_BASE_URL
  fallback       PASS   unknown agent, no --provider -> anthropic passthrough (not autodetected)
  explicit       PASS   unknown agent + --provider openai -> openai wire (universal recipe)

OK — 4 autodetected entry point(s) + 2 universal-recipe invariants reproduced (browserless)
```

The exact spacing may vary by terminal width. The invariant is that every known entry point
passes and the two universal fallback/explicit-provider recipes pass.
