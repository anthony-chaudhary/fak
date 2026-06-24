# extdriver — captured run

Captured output of `cd examples/extdriver && go run .` (this is a separate Go module;
the root build does not descend into it). The run is **deterministic** — every
invocation prints exactly this, with no model, network, or randomness — and `main`
exits **non-zero** if any leg fails, so the exit code gates correctness in CI.

```
$ cd examples/extdriver && go run .
fak out-of-tree driver — ABI importable via pkg/abi
  ABI version: v0.1

[1] claimed OpCode 65578 (OpsVendor range [65536,131072))
    registered without a clash -> opcode is claimed

[2] claimed VerdictKind 1031 (VerdictsVendor range [1024,65536))
    FoldRank(1031) = 200 -> kind is registered in the frozen lattice

[3] Adjudicate(tool="refund_payment") -> Kind=1031 Reason=POLICY_BLOCK By=extdriver/denyTool
    Adjudicate(tool="read_file") -> Kind=5 (VerdictDefer) — fold identity holds

extdriver: OK — both vendor numbers claimed and the driver round-trip passed
```

The final `extdriver: OK` line and a zero exit status are the witness: both vendor
numbers were claimed and the `Adjudicator` round-trip held (a clash on either number,
or a wrong verdict, would have exited non-zero and failed CI).
