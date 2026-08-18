# Paired harness-control task

Produce a customer-support harness for Linux/amd64 contract `v1` with these effective behaviors:

- component `kernel@1.0.0` provides `runtime`;
- `instruction:response-style` is `detailed`;
- `tool:search_kb` is available;
- policy `tools` grants `search_kb` and denies `shell`;
- the effective product has a cryptographically verified lock;
- a consequential-change preview exists before admission; and
- a runtime observation verifies the final instruction, tool, and policy against the final lock.

The default-control arm receives a verified starting product whose response style is `concise`; all other required behavior is present. The scratch arm receives no product manifest or lock.

Stop the clock after all of these exist in the arm directory:

1. final verified lock;
2. human-readable effective capability inventory with provenance;
3. consequential-change preview (default-control) or equivalent before/admission comparison (scratch);
4. runtime observation and successful lock-to-run verification;
5. exact command transcript and artifact hashes.

Do not add unrelated providers, UI, agents, or deployment packaging.
