# Captured component compatibility receipt

Command, run from the repository root on 2026-08-20:

```powershell
./examples/component-contracts/run.ps1
```

Output:

```text
ALLOW stack for component-compatibility-check
selected: 3 components
  - cache.kv.radix@1 (cache; cache release 1.0.0)
  - kernel.paged-attention@2 (kernel; kernel release 2.1.0)
  - runtime.cuda@12.9 (runtime; CUDA runtime 12.9 installation receipt)
  cache.kv.radix@1 requires -> kernel.paged-attention@2 [cache release 1.0.0 compatibility suite]
  kernel.paged-attention@2 requires -> runtime.cuda@12.9 [kernel release 2.1 device witness]
WARN RECOMMENDATION_UNMET: cache.kv.radix@1 wants runtime.cuda.graphs — recommendation is not a launch requirement [cache benchmark 42]
```

Exit code `0` means the local hard-requirement chain resolved and the expected
missing recommendation stayed non-blocking. The runner exits nonzero if either
part of that receipt changes.
