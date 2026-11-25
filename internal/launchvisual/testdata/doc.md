# Fixture launch note

Decode reaches 118.1 tok/s in the same window, against 81.0 for the stock build.

| Row | Value |
| --- | --- |
| same window fak | `hl:samewindow.fak.decode_tok_s` 118.1 |
| samewindow stock | `hl:samewindow.stock.decode_tok_s` 81.0 |
| fork fak | `hl:fork.parity.fak_tok_s` 119.9 |
| fork realugbun | `hl:fork.parity.fork_tok_s` 119.5 |
| spec plain | `hl:spec.plain.tok_s` 119.7 |
| spec depth 2 | `hl:spec.d2.tok_s` 171.5 |
| spec depth 4 | `hl:spec.d4.tok_s` 180.9 |
| climb first native | `hl:climb.first_native.tok_s` 6.32 |
| climb resident | `hl:climb.resident.tok_s` 82.5 |
| climb fused box 2 | `hl:climb.fused_box2.tok_s` 101.6 |
| c16 true batch | `hl:c16.true_batch.decode_tok_s` 205.67 |
| oss c16 decode | `hl:oss.c16.decode_tok_s` 402.76 |

The opt-in speculative path reaches 171.5 tok/s, a 1.43x ratio on the same receipt.

The build took 12.5 seconds on the fixture box.

```
an unchecked figure 9.99 inside a fence
```

See https://example.com/docs/1.25 and version v0.58.0 for details.
