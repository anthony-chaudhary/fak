# Guard remote maintenance

Remote maintenance is a narrow operator-reviewed authority for one typed tool,
`fak_remote_maintenance`. It does not expose SSH, a shell, a caller-selected
command, or caller-selected paths. Tool input is one strict JSON object:

```json
{"profile_id":"reviewed-profile","operation_digest":"<lowercase sha256>"}
```

Extra fields are refused. The only v1 operation is
`systemd-profile-install-reload`. Its native adapter exchanges the fixed
`fak.remote-maintenance.request.v1` and `fak.remote-maintenance.response.v1`
envelopes for resolve, validate, and execute. The resulting verified receipt is
`fak.remote-maintenance.receipt.v1`.

## Operator action

An operator reviews a complete config out of band, including target, pinned
OpenSSH host-key fingerprint, operation, payload digest, exact paths, staging
prefix objects (`{"directory":"/var/tmp","filename_prefix":"maintenance-stage."}`),
units, lease nodes and holder, expiry, and a native executable plus immutable
SHA-256. Prefix objects are sorted by directory then filename prefix; a bare
directory string is not a valid scope. Interpreters and arbitrary shell payloads
are refused.

Install it for a short window:

```text
fak manage allow --remote-maintenance REVIEWED_CONFIG --ttl 15m
```

The TTL is mandatory, positive, and at most 15 minutes. Installation validates
every profile and executable, clamps every reviewed expiry to the shorter of its
existing expiry and the requested TTL, and atomically writes the private user
sidecar at `~/.fak/guard/remote-maintenance.json`. The command installs authority;
it never performs maintenance.

Revoke one profile with:

```text
fak manage allow --remove --remote-maintenance PROFILE_ID
```

Removing the last profile removes the sidecar. Expiry, deletion, unreadable or
mutated config, changed executable bytes, host/boot/prior-state drift, or stale
lease evidence all disable remote maintenance while leaving the base guard/serve
floor running. An expired remaining profile cannot prevent revocation. If the
sidecar is too malformed to identify a profile, the operator can remove all remote
authority with `fak manage allow --remove --remote-maintenance '*'`. Normal
`fak manage --guard` and `fak serve` load the sidecar; the sidecar path is
SELF_MODIFY-protected even while absent, and configured adapter executables are
protected when present.

The legacy allow overlay and exact-call `--approve` path are unchanged. A remote
maintenance grant cannot bypass a kernel deny, peer lease, or another hard floor.
The authority revalidates the same resolved grant before execution and again at
its transaction boundary; no request metadata supplies authority.
