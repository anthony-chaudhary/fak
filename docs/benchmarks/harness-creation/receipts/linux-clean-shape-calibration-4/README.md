# linux-clean-shape-calibration-4

Maintainer calibration only; it is excluded from independent promotion evidence.

- Generator: `github.com/anthony-chaudhary/fak/cmd/fak@83c7641b909e`
- Environment: WSL Ubuntu 24.04, Linux amd64, empty `HOME`, `GOPATH`, `GOCACHE`, and `GOMODCACHE`
- Outcome: success in 79.873 seconds
- Install: 42.019 seconds; product build: 27.249 seconds; selfcheck: 0.002 seconds; rerun: 0.019 seconds
- User file SHA-256 before and after rerun: `9b96b0bc92c757958686e096ea6e080f005d42cfd3ad856faf47ad083fc5a92e`
- Closed transcript SHA-256: `6e9e3232295eb6d8db5606b4e40971f4669fcab98e220e81bd109fb0c464fd88`

The `transcript_sha256` line inside the raw transcript is a prefix hash computed while
`tee` still had the file open. The checksum above was computed after the runner exited and
is authoritative. `run.sh` labels future in-stream values `transcript_prefix_sha256` to
make that distinction explicit.

The natural user-owned replacement is the shape that failed before #6940. This run proves
the shape-safe fix works; it does not prove unfamiliar-builder usability or freeze a tuned
baseline.
