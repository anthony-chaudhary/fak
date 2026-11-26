# piidemo example output

Captured from `go test ./cmd/piidemo -run 'TestSelfcheck$' -count=1 -v`. This runs
the same headless witness as `go run ./cmd/piidemo -selfcheck`: the shared
PII-bearing scenario is replayed against both arms over the real gateway HTTP
seams, and each assertion prints one PASS line. The planner is deterministic, so
the output reproduces line for line.

```
=== RUN   TestSelfcheck
    piidemo_test.go:17: PASS raw.messages_200
    piidemo_test.go:17: PASS raw.zero_admissions
    piidemo_test.go:17: PASS raw.no_fak_note
    piidemo_test.go:17: PASS guarded.messages_200
    piidemo_test.go:17: PASS guarded.transform_verdicts
    piidemo_test.go:17: PASS guarded.fak_note
    piidemo_test.go:17: PASS guarded.pii_masked
    piidemo_test.go:17: PASS guarded.secret_masked
--- PASS: TestSelfcheck (0.04s)
PASS
ok  	github.com/anthony-chaudhary/fak/cmd/piidemo	0.087s
```

The raw assertions show the default arm forwarding the tool output byte-identical
with zero admissions and no note. The guarded assertions show the masked spans,
the in-band `[fak]` note, and the admitted bytes. The `(0.04s)` timing and the
`ok` line vary by machine; the PASS lines do not.
