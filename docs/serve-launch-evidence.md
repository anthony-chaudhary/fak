# Serve launch evidence

`fak serve --launch-evidence /absolute/new/launch.json` writes an opt-in private
startup snapshot after model and gateway setup, before listener execution. The
parent directory must exist. Each startup requires a new path; an existing file
or symlink refuses startup. Partial output from a failed write remains evidence
of uncertainty and must not be accepted.

The `fak.serve_launch_evidence.v1` snapshot includes PID, systemd invocation ID,
executable SHA256, embedded Go VCS identity, the SHA256 of the exact configuration
bytes parsed at startup, and an allowlist of resolved controls. No raw arguments,
environment, configuration content, upstream URLs or credentials are emitted.
Missing VCS or invocation identity remains empty; consumers must refuse it when
process attestation requires those fields. A missing loaded native model reports
backend `unresolved`. The prefill chunk count is a configured ceiling.

A collector must compare PID, invocation and executable SHA256 with a fresh,
stable live process observation before trusting the snapshot. It must also
validate source identity and control bounds. Startup evidence does not prove
listener readiness, successful inference, model byte identity, active forward
path, achieved chunk size, token throughput, or hardware qualification.
