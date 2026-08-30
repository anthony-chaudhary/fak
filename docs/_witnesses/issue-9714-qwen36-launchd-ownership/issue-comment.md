Current ownership-only verdict: **HOLD** (`alternate_launchd_supervisor_owns_incumbent`).

- Read-only preflight found the expected `com.fak.qwen36-model` job and its LaunchAgent plist absent. The single healthy port 8090 listener is not bound to that expected identity.
- The incumbent still matches preserved command digest `a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d`; its alternate owner matches public-safe label digest `b567298df044cecdac3cf921d7cb971e4665db7b5407fb634647a910e3abbdb7`.
- Ninety-one read-only samples spanning 109 seconds retained one listener, both HTTP 200 responses, and alias `qwen3.6-27b` with unchanged command and owner digests.
- No GPU lease, signal, bootout, bootstrap, service mutation, Qwen3.8 artifact access, or model arm occurred. External service state was unchanged and healthy at exit.

The current packet does not authorize migration or bootout of the proven alternate owner, and the expected service definition is absent. The next route is an explicit service-owner decision supplying the reviewed `com.fak.qwen36-model` definition and migration authority; then repeat the canonical lease-held TERM-only drill and require exact PID/command binding plus 90 seconds of post-restore stability.
