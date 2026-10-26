# Guard remote-maintenance independent review — 2026-10-25

ACCEPT [SW-VERIFIED] for frozen scope `34180f4c943829a9c09db73d1a3e5aaebc4aad9c8dd6eaa5a27d86ffc3541bd6` against public core `eddf9baa062655d0ea932db5ef25cc8dadd51b6b`. An independent coordinator reviewed the implementation and ran the commands below. All 19 source SHA-256 values matched the frozen manifest before and after execution; no implementation file was changed by this reviewer.

The registered default-policy MCP path executes a pinned native authority once. Caller-selected authority is refused. Alias admission is retained before canonical resource checks enforce credential, self-modification and egress floors. Computed shell flags and caller read-only hints do not admit operator context installation. Invalid or expired remote context leaves the base guard running. Execution uncertainty remains a bound, unverified receipt; it is never reported as a pre-effect denial or replayed.

| Independently executed command (GOWORK=off, GOMAXPROCS=2) | Result | Log SHA-256 |
| --- | --- | --- |
| `go test -json -p 1 ./internal/gateway -run RemoteMaintenance -count=1 -timeout 90s` | PASS; 49 test/subtest pass events, no failures | `193ba5df4ca4d8453c271913cb6d36afa813342f36cacb6d43e68069b704057c` |
| `go test -json -p 1 ./cmd/fak -run RemoteMaintenance -count=1 -timeout 90s` | PASS; 10 test/subtest pass events, no failures | `3b8d400c332e0095e4bf5c953ef2b002ead64bbfa943db86df4b33524503529b` |
| `go test -json -p 1 ./internal/adjudicator -run CLIGrammarReadOnly -count=1 -timeout 90s` | PASS; 1 test, no failures | `fce4c4114f80c11c73d823d2e65103dc0689cf669ca0baedc6a6d32b225a196e` |

The native adapter tests use isolated temporary state and test executables. This review installs no operator context, approves no call, performs no SSH maintenance, and makes no hardware, reboot, live-bridge deployment or full-suite claim. Actual operator authorization and live entrypoint verification remain required. Any change to the file set below invalidates this acceptance.

## Frozen implementation hashes

| File | SHA-256 |
| --- | --- |
| `cmd/fak/guard_allow_approve_test.go` | `a2ef559b6215f535613a10cf894b17159f5f2d198ebf1e021631ee934212b87b` |
| `cmd/fak/guard_allow.go` | `028ed219b6f4aafadc68e47f5b5949dce6afe1ece06f307e8de27a35770e398a` |
| `cmd/fak/guard_remote_maintenance_test.go` | `11f90506b43028f7595f39bda6d795e4331d6e6563cbdd686ae0e4f157bb02e3` |
| `cmd/fak/guard_remote_maintenance.go` | `c4df22d5e8882530e7166dbc4fa2ed743aa875b3d31a75f63f297f9a8fca2f60` |
| `cmd/fak/guard_startup.go` | `40f5aa62f4d561a6fb185e202a5e0148c321219113f446724a51e8a22901f3b6` |
| `cmd/fak/guard-default-policy.json` | `c3d249d09c3f6f5c47700c9e200d2fb520c14c10bb72762ef5fa091e366c7902` |
| `cmd/fak/guard-strict-policy.json` | `68fc6dc651550fc6b518332f4b463c7284fa4b46eabf40db78ad30e1cb8f5f4d` |
| `cmd/fak/leaseplane_endpoint_test.go` | `f54eb8a57ddf84f34dfa6cb12af988888a61bb35dec8e1d72df7bca38655e95f` |
| `cmd/fak/main_policy.go` | `669e0d9208e4f52985e8988cf5fea2e690d16c519744158ab52b1594f7cafe3d` |
| `docs/notes/GUARD-REMOTE-MAINTENANCE.md` | `3d066cb4594230fe18f3e15c423aee8b7c6ca1559b8de39a782523a1d1f86ceb` |
| `internal/adjudicator/decide.go` | `874925a67fb4c234225265a771dddb4f510fcca50ddae06b18cdaf54d52827c6` |
| `internal/adjudicator/remote_maintenance.go` | `d1790b0201901a0029c88532cb531a856253d24e1cac46f65003134d9da7f5ce` |
| `internal/gateway/coretool_admit.go` | `39a1b9a40ca8e70809fb2e339a5c91d096ac95afd3b4a6b8fbd7fb97ef39ba5e` |
| `internal/gateway/mcp.go` | `70e101102afd988f95c4fcaa48ba8a9a0e57a60263e0d7668c7c237cacdb9a04` |
| `internal/gateway/remote_maintenance_test.go` | `d561bcf75661be629c1e25074c046e0114b1e26e0e80c33cbe8a246bb7f04c6c` |
| `internal/gateway/remote_maintenance.go` | `b45b620b87148d52dcb39174f75c9347b556986c229a60affbb13d181ce4ab8e` |
| `internal/gateway/server_state.go` | `9de707ba0a2fa12cefb35c804c17ba8e46357616646b2d5b962a90b2a27351e6` |
| `internal/gateway/tool_routing_lease_admission.go` | `7d19d0cf5dc5345784c26cd8f8dc66f8b7f5006e45dab0b5b02d7d1c184a17ee` |
| `internal/gateway/tool_routing.go` | `7fc2a15248fbf433de9c1cfe354fda45ef4ad2769e28de2b4f41dd835f83062f` |
