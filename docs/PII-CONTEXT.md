---
title: "Context-aware PII handling"
description: "Documentation for Context-aware PII handling, including the captured behavior, operating context, and reproducible fak evidence."
---

# Context-aware PII handling

fak's lexical PII detector identifies **shapes**, not the meaning or policy of a value.
An email-shaped value is therefore protected by default. In contexts where a class is
intentionally public (for example, a tool that returns published recruiting contacts), the
trusted caller can classify that class as public on the `abi.ToolCall`:

```go
call.Meta["fak.pii.public_classes"] = "email"
```

The value is a comma-separated list from this closed taxonomy:

- `email`
- `phone`
- `national_id`
- `payment_card`
- `iban`

This is class-scoped rather than a global PII bypass. If the result contains another class,
fak still masks it under the default warn posture or seals it under fail-closed posture.
Unknown class names, malformed lists, result-authored metadata, and obfuscation that cannot
be safely located all fail closed.

## General diagnosis pattern

When a legitimate field is replaced by `[redacted:pii:<n>B]`:

1. Treat the sentinel as evidence that a boundary transform ran; do not put it into an
   application field or persist it as source data.
2. Separate **shape detection** ("this looks like an email") from **context policy** ("emails
   returned by this trusted tool are public here").
3. Put the narrow classification on the caller/request boundary, never in untrusted result
   content and never as a detector-wide disable.
4. Verify a mixed payload: the public class must survive byte-for-byte while an undeclared
   class is still masked or sealed.
5. Reject unknown taxonomy values so configuration mistakes cannot broaden access.

This pattern generalizes to future false positives: retain conservative detection, add a
closed semantic class, grant the smallest context-specific exemption at a trusted boundary,
and prove both the intended pass case and the protected mixed case.
