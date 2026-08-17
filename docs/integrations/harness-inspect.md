# Inspect a harness before trusting it

Most people should not need to build an agent harness from scratch. Start from a useful resolved product, then **trust but verify** its effective behavior before launch.

`fak harness inspect` verifies the cryptographic ID in a resolved harness lock and turns that lock into an operator-first inventory:

- every runtime component, where it came from, and why the resolver chose it;
- every effective instruction, tool, memory, policy, route, secret, workflow, and UI asset;
- the layer that supplied each capability instead of an opaque label such as “skill”;
- whether that capability is changeable, locked by its source, or mandatory;
- the environment and resource budget; and
- concrete commands for comparing and verifying a proposed change.

```bash
fak harness inspect --lock product.lock.json
```

The output begins with `HARNESS INSPECT | VERIFIED`. Verification fails before anything is rendered if the lock ID does not match its contents.

Use JSON when another control surface needs the same evidence:

```bash
fak harness inspect --lock product.lock.json --json
```

## Control without reconstruction

Inspection is read-only. To change an existing harness:

1. edit the product manifest or selected contextual layers;
2. run `fak harness resolve` to produce a candidate lock;
3. run `fak harness preview --current product.lock.json --candidate candidate.lock.json` to review only consequential changes; and
4. run `fak harness inspect --lock candidate.lock.json` to verify the whole resulting harness, not just the diff.

This separates three questions that opaque defaults tend to conflate:

- **What is actually active?** `harness inspect`
- **What would change?** `harness preview`
- **What am I allowed to alter?** the `changeable`, `locked by source`, and `mandatory` control labels

The lock remains the machine contract. Inspection is the human-readable trust boundary over that contract, not a second source of truth.
