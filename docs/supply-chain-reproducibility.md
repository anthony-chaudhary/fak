# Supply-chain posture: reproducible builds

Status note for #3711 (epic #3708). This is the honest statement of what is
byte-reproducible, how CI witnesses it, what is documented intent only, and what
is explicitly out of the guarantee.

## What is reproducible

The **pure-Go targets** — the static `CGO_ENABLED=0` binary the 5-target release
matrix ships and the distroless image wraps. Byte-identity rests on:

- `CGO_ENABLED=0` — no host C toolchain can leak into the artifact;
- `-trimpath` — no host build path is embedded;
- `go.mod`/`go.sum` hash determinism — inputs are content-addressed;
- the Go toolchain itself: Go ≥1.21 binaries embed **no wall-clock build
  timestamp**, so no clock input is needed for the binary to reproduce.

`internal/architest` enforces that any `import "C"` file is fenced behind an
opt-in build tag, which is the invariant that keeps the **default** build pure-Go
— and therefore *able* to be reproducible at all.

All shipping consumers build through the one canonical recipe,
[`scripts/build.sh`](../scripts/build.sh) (#3709) — the release matrix
(`.github/workflows/release-artifacts.yml`), both Dockerfiles, and the local
`make build`/`release` targets. `tools/build_entrypoint_test.py` reds if the flag
string reappears inline in any consumer.

## How it is witnessed

[`.github/workflows/reproducible.yml`](../.github/workflows/reproducible.yml)
builds the pinned linux/amd64 release target **twice** — each build from its own
detached git worktree (two distinct absolute build paths, exercising `-trimpath`)
with its own `GOCACHE` (so the second build re-derives every byte instead of
replaying the first's object cache) — via the same `scripts/build.sh` recipe the
release matrix uses. It emits both SHA-256 sums to the job log and step summary,
surfaces the recorded build settings (`go version -m`, so a rebuilder can confirm
toolchain/flags/VCS revision), and **fails** if the sums differ. It runs
paths-filtered on pushes/PRs touching the build inputs, plus a weekly cron
backstop.

## Documented intent only (not witnessed here)

`SOURCE_DATE_EPOCH` (commit-derived, e.g. `git log -1 --format=%ct`) would make
the **archive/container layer** — tar mtimes, image timestamps — deterministic.
The pure-Go binary does not consume it, and the double-build check builds no
tar or image, so there is no witnessed artifact for it here. It is recorded as
intended posture for the archive/image path, owned alongside the product-images
(#3256) and air-gapped-kit (#3279) work.

## Explicitly out of the guarantee

The `Dockerfile.cuda` path (`CGO_ENABLED=1 -tags cuda`, nvcc-linked
`libfakcuda.a`). Its build stage compiles against the CUDA `-devel` base image
(`nvidia/cuda:12.6.2-devel-ubuntu22.04`), and that C toolchain is **not pinned**
— so, mirroring Hugo's "extended needs a pinned C toolchain" caveat, the cuda
image is not claimed reproducible until it is. The default distroless-static
image and the 5 release targets are what the byte-identity check covers.

## How to independently rebuild and compare

```sh
git checkout <the release tag>
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 \
  OUT=/tmp/fak-rebuilt VERSION="$(cat VERSION)" sh scripts/build.sh
sha256sum /tmp/fak-rebuilt      # compare against the published SHA256SUMS entry
go version -m /tmp/fak-rebuilt  # confirm recorded toolchain/flags/VCS inputs
```

Any Go toolchain matching `go.mod`'s directive (setup-go resolves the same way in
CI) on any host OS should produce the identical linux/amd64 bytes.
