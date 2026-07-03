# mobile-ffi — captured run

Captured output of `go -C examples/mobile-ffi run .` (a separate Go module; the
root build does not descend into it). The run is **deterministic** — every
invocation prints exactly this, with no model, network, or randomness — and
`main` exits **non-zero** if any leg is wrong, so the exit code gates
correctness. This is the pure-Go witness of the SAME deny/allow round-trip the C
shim (`libfakmobile.go`) carries to Android/iOS.

```
$ go -C examples/mobile-ffi run .
fak mobile FFI — on-device tool calls routed through the adjudicator floor
  ABI version: v0.1

[1] dangerous {"tool":"send_sms","allow":false,"verdict":"DENY","reason":"POLICY_BLOCK","by":"mobile/floor"}
[2] benign {"tool":"get_battery_level","allow":true,"verdict":"ALLOW","reason":"","by":"mobile/floor"}
[3] unknown  {"tool":"transfer_funds","allow":false,"verdict":"DEFAULT_DENY","reason":"DEFAULT_DENY","by":"mobile/fold"}

mobile-ffi: OK — dangerous denied, benign continued, unknown failed closed
```

The three legs are the acceptance shape (a denied dangerous call + a continued
benign one, mirroring the desktop live pilot), plus the fail-closed default for
an unrecognized call. A wrong verdict on any leg exits non-zero.
