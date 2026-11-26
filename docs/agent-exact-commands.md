---
title: "Grant one exact command to the native agent"
description: "Use a narrow, repeatable command grant without widening the native Bash allowlist."
---

# Grant one exact native-agent command

Native and interactive `fak agent` runs use default-open Bash under the kernel policy.
The kernel still evaluates explicit denials. Select `--bash-policy focused` to
also check commands against the focused list. The default `auto` mode selects
focused Bash for exact grants, a policy manifest, or the `fail_closed` posture.
Select `--bash-policy default_open` to use kernel admission with an exact grant
present. Raw and dual runs keep focused Bash.

In focused mode, grant an additional operator-reviewed command byte for byte:

```text
fak agent --native \
  --allow-bash-command "my-tool inspect --json" \
  --task "Inspect the workspace with the approved command and summarize the result."
```

Repeat `--allow-bash-command` to grant more than one command. Each value is an
exact match: extra arguments, shell operators, prefixes, suffixes, or different
spacing produce a different command and remain refused. Empty grants are invalid.

The grant applies to this process. Kernel adjudication, workspace checks and
command timeouts still apply. Code tools are enabled by default. A grant requires
`--code-tools=true`.

Each Bash call has a two-minute default deadline. For an approved command that
needs longer, set a positive process-local override up to ten minutes:

```text
fak agent --native \
  --allow-bash-command "my-tool verify --json" \
  --bash-command-timeout 8m \
  --task "Run the approved verification command and report its result."
```

The override changes only the Bash process deadline. It does not widen the
command grant or any other code-tool limit. Supplying it with
`--code-tools=false` is an error.

For the server-owned native `POST /v1/messages` agent loop, arm the same
byte-exact command on the `fak serve` process:

```text
fak serve --native --native-code-tools \
  --native-allow-bash-command "my-tool verify --json" \
  --native-bash-command-timeout 8m
```

Repeat `--native-allow-bash-command` for additional commands. Server-side
grants require native code tools and preserve focused defaults and every other
gate. The server timeout has the same two-minute default and positive ten-minute
maximum. These grants apply only to the server-owned `/v1/messages`
`agent.RunArm` loop. Chat/proxy adjudication keeps its independent unknown
write-footprint and lease refusal; this flag does not grant authority across
that boundary. Client-side grant and timeout values are process-local and are
not transmitted over HTTP.
