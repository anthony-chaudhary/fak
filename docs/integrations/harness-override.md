# Control a default harness without rebuilding it

`fak harness inspect` answers what is active. `fak harness override` turns one **changeable** inspection row into a concrete, reviewable layer proposal.

```bash
fak harness inspect --lock product.lock.json
fak harness override \
  --lock product.lock.json \
  --capability instruction:response-style \
  --value detailed \
  --output operator-override.json
```

The command verifies the current lock, requires the exact `kind:id` shown by inspection, and refuses capabilities marked `locked by source` or `mandatory`. The generated file uses the existing `fak.harness-assets/v1alpha1` composition contract; it is not a second customization system.

For policies, overrides are narrowing-only:

```bash
fak harness override \
  --lock product.lock.json \
  --capability policy:tools \
  --deny shell \
  --output operator-override.json
```

Policy grants cannot be added through this path. Existing composition checks reject widening when the generated layer is resolved.

## Review before admission

Append the generated layer to the product manifest after the capability's current source, include `operator-override` in the selection, and resolve a candidate lock. Then use both visibility seams:

```bash
fak harness preview \
  --current product.lock.json \
  --candidate candidate.lock.json

fak harness inspect --lock candidate.lock.json
```

Preview names an effective instruction, memory, route, workflow, or UI value change as `behavior-change`, with the generated layer as provenance. Inspection then shows the complete candidate rather than only its diff.

This is intentionally proposal-first: the command does not silently mutate the current manifest or admit the candidate. Operators get control, while lock verification and consequential-change review remain explicit checkpoints.
