# Installing `fak`

`fak` runs local models and governs agent tool calls. Public installation defaults
prioritize useful GPU builds: Apple Silicon Metal, Linux amd64 Vulkan (including
AMD Strix Halo), and the NVIDIA CUDA container. CPU builds are secondary references
available through explicit selection. This page is for adopters installing without
cloning the repository.

Three supported paths, fastest first:

1. [One-line installer](#1-one-line-installer-recommended) — `curl | sh`, downloads a
   verified prebuilt binary.
2. [Manual download](#2-manual-download) — grab the archive for your OS/arch from the
   GitHub release yourself.
3. [Docker](#3-docker) — the NVIDIA CUDA image, or a CPU reference image.

[What's in the binary](#whats-in-the-binary) answers the question the three paths above
skip — why one ~13 MB binary carries hundreds of verbs, and which of them are yours. A
[build-from-source](#build-from-source) fallback and the [`go install`
status](#about-go-install) are at the end.

Next: once `fak` is on your PATH, the [first-session tutorial](docs/fak/tutorial.md)
walks you from the binary to your first adjudicated tool call — fully offline, no key or GPU.

The GPU archive contract is `darwin/arm64` Metal and `linux/amd64` Vulkan.
Vulkan archives require GNU glibc 2.39+ (Ubuntu 24.04+), the system Vulkan
loader (`libvulkan.so.1`), and a compatible GPU driver. They include a `spirv/`
directory that must stay beside the binary. The installer rejects older glibc;
build Vulkan from source for older distributions.
CPU reference archives remain available for `linux/amd64`, `linux/arm64`,
`darwin/amd64`, `darwin/arm64`, and `windows/amd64`.

These GPU-first archive changes take effect in a release carrying those assets;
historical v0.54.0 has neither GPU archive. Until such a release is available,
build the selected backend from source or explicitly request a CPU reference.
A packaged backend still needs qualification on the actual model and GPU.

---

## 1. One-line installer (recommended)

```sh
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

The installer ([`install.sh`](install.sh)) defaults to Metal on Apple Silicon and
Vulkan on Linux amd64. If `nvidia-smi -L` detects NVIDIA, it prints the CUDA
container command. Other platforms require explicit `--variant cpu` for a CPU
reference. Missing GPU archives never silently fall back to CPU.

Downloads are verified against `SHA256SUMS` before installation. Vulkan shader
resources are installed to `spirv/` beside `fak`. The installer needs POSIX `sh`,
`curl` (or `wget`), `tar`, and `sha256sum` (or `shasum`).

```sh
# Explicit CPU reference:
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh -s -- --variant cpu
```

Then confirm:

```sh
fak version          # prints the installed version, e.g. 0.55.0
```

What you'll see: a single version line on stdout (the release you just installed). If
`fak version` resolves and prints, the binary is on your PATH and the install worked.

> **`command not found` right after install?** On a fresh macOS, `/usr/local/bin` is not
> writable without `sudo`, so the installer falls back to `~/.local/bin` — which is **not**
> on the default PATH. Add it (zsh is the macOS default shell):
> `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && exec zsh`. Or install
> straight to a dir already on PATH: `FAK_INSTALL_DIR=/usr/local/bin sudo sh -c 'curl -fsSL .../install.sh | sh'`.

Knobs (environment variables):

| Variable | Effect | Default |
| --- | --- | --- |
| `FAK_VERSION` | Pin a version, e.g. `0.55.0` | latest release |
| `FAK_INSTALL_DIR` | Install target directory | `/usr/local/bin` if writable, else `~/.local/bin` |
| `FAK_REPO` | `owner/repo` override | `anthony-chaudhary/fak` |

Example — pin a version into a user-local dir:

```sh
FAK_VERSION=RELEASE_WITH_GPU_ASSETS FAK_INSTALL_DIR="$HOME/.local/bin" \
  sh -c 'curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh'
```

> Piping a script into a shell runs it with your privileges. If you'd rather read it
> first, download `install.sh`, inspect it, then run `sh install.sh`.

---

## 2. Manual download

If you don't want to run an installer, take the archive straight from the
[Releases page](https://github.com/anthony-chaudhary/fak/releases). Assets are named:

```
fak_<version>_darwin_arm64_metal.tar.gz   # Apple Silicon GPU
fak_<version>_linux_amd64_vulkan.tar.gz   # Linux GPU, includes spirv/
fak_<version>_<os>_<arch>.tar.gz          # CPU reference
fak_<version>_<os>_<arch>.zip        # windows
fak_<version>_<os>_<arch>.tar.gz.sha256
SHA256SUMS                           # aggregate, all targets
```

### Linux / macOS

```sh
VERSION=RELEASE_WITH_GPU_ASSETS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')          # linux | darwin
ARCH=$(uname -m); [ "$ARCH" = x86_64 ] && ARCH=amd64; [ "$ARCH" = aarch64 ] && ARCH=arm64
case "${OS}/${ARCH}" in
  darwin/arm64) BACKEND=metal ;;
  linux/amd64) BACKEND=vulkan ;;
  *) echo "Choose a supported GPU target or an explicit CPU reference archive"; exit 1 ;;
esac
ARCHIVE="fak_${VERSION}_${OS}_${ARCH}_${BACKEND}.tar.gz"
BASE="https://github.com/anthony-chaudhary/fak/releases/download/v${VERSION}"

curl -fsSLO "${BASE}/${ARCHIVE}"
curl -fsSLO "${BASE}/SHA256SUMS"

# Verify before trusting the binary.
grep " ${ARCHIVE}\$" SHA256SUMS | sha256sum -c -

# Stronger than the checksum: verify the SLSA build-provenance attestation — proof
# the archive was built by this repo's release workflow from a tagged commit
# (requires the `gh` CLI). Each release asset is attested by release-artifacts.yml.
gh attestation verify "${ARCHIVE}" --repo anthony-chaudhary/fak

tar -xzf "${ARCHIVE}"        # extracts: fak, LICENSE, GETTING-STARTED.md
chmod +x fak
sudo cp fak /usr/local/bin/  # or any dir on your PATH
# Vulkan requires its bundled resources beside fak:
if [ "$BACKEND" = vulkan ]; then sudo mkdir -p /usr/local/bin/spirv; sudo cp -R spirv/. /usr/local/bin/spirv/; fi
fak version
```

### Windows (PowerShell)

This is the secondary CPU reference build.

```powershell
$Version = "0.55.0"
$Archive = "fak_${Version}_windows_amd64.zip"
$Base    = "https://github.com/anthony-chaudhary/fak/releases/download/v$Version"

Invoke-WebRequest "$Base/$Archive" -OutFile $Archive
Invoke-WebRequest "$Base/SHA256SUMS" -OutFile SHA256SUMS

# Verify
$want = (Select-String " $Archive$" SHA256SUMS).Line.Split()[0]
$got  = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower()
if ($want -ne $got) { throw "checksum mismatch for $Archive" }

Expand-Archive $Archive -DestinationPath fak-dist
# Move fak-dist\fak.exe somewhere on your PATH.
.\fak-dist\fak.exe version
```

The downloaded binary carries its version stamped at build time, so it reports the right
version even with no `VERSION` file alongside it.

---

## 3. Docker

For NVIDIA native inference, start with the GPU image on a host with NVIDIA
Container Toolkit configured:

```sh
docker run --rm --gpus all ghcr.io/anthony-chaudhary/fak:cuda-latest version
```

Then run `serve` with a supported model mounted into the container. A version
check confirms installation; it does not qualify inference on physical hardware.

The root [`Dockerfile`](Dockerfile) builds the secondary CPU reference/governance
image (`CGO_ENABLED=0`, distroless). For a gateway in front of another engine:

```sh
docker build -t fak .

# Front a model served by Ollama on the host:
docker run --rm -p 8080:8080 fak serve --addr 0.0.0.0:8080 \
    --base-url http://host.docker.internal:11434/v1 --model qwen2.5:1.5b
```

Containers must bind `0.0.0.0`, not loopback — the default `CMD` already does
(`serve --addr 0.0.0.0:8080`). Stamp a specific version into the binary at build time:

```sh
docker build --build-arg APP_VERSION=0.55.0 -t fak:0.55.0 .
```

Override the entrypoint command to run `fak agent`, `fak policy`, etc. instead of the
gateway.

---

## What's in the binary

One `fak` binary dispatches roughly **270 verbs**. Two dozen are the product you installed;
the large majority exist to develop *this repository* — commit gating, lane leases, fleet
dispatch, scorecards, release plumbing. That is startling if you came for a gateway, so
here is the split. It is not a convention you have to learn: the binary carries it as data.

| Tier | Count on `main` | What it is | Lists itself with |
| --- | --- | --- | --- |
| **frontdoor** | 26 | the product: gateway, guard, policy, sessions, models, audit, attest | `fak help` |
| *hidden* | 6 | re-exec / hook seams `fak` spawns for itself; never listed | — |

Don't trust those counts — they move as verbs land, and none of the rest of this page
depends on them. Repository-development commands are not part of this runtime count; maintainers install the separate `fak-dev` artifact with `go install ./cmd/fak-dev`, and `fak-dev help` lists that surface. Derive your own runtime count: `fak help --all` prints the whole list
for the build you actually installed, `fak help --all` prints every listed verb tagged with
its tier, and `fak-dev index verbs --json` emits the same classification as machine-readable
rows. The classification has exactly one home in the source
([`internal/devindex/tiers.go`](internal/devindex/tiers.go)) and a test refuses a newly
dispatched verb that no tier claims, so a verb cannot drift into the wrong column.

### Is the dev tier a problem in a production image?

No — and the reason is specific, not reassurance:

- **A verb runs only when it is the command you invoke.** Nothing in the dev tier is
  scheduled or started by `fak serve`, and no HTTP or MCP request can reach one: the
  gateway never spawns a process.
- **The container never invokes one.** The [`Dockerfile`](Dockerfile) sets
  `ENTRYPOINT ["/usr/local/bin/fak"]` with `CMD ["serve", "--addr", "0.0.0.0:8080"]`.
- **Most dev verbs have nothing to act on outside a checkout.** They locate the repository
  by searching upward for `dos.toml` / `go.mod`; a distroless image has neither.
- **They cost bytes, not behaviour** — and the bytes are the ~13 MB the Docker section
  above already quotes.

Whichever verb you run, the one thing `fak` writes unprompted is a local usage journal in
`<user-config-dir>/fak/` (`usage.jsonl` plus the `usage.salt` it hashes with): verb name,
the *number* of arguments, a salted digest of the argument vector, exit code, duration,
version, hostname, pid — never argument values, and never off the machine. Set
`FAK_USAGE_LOG=off` to disable it. The full account — why `off` is the only spelling that
disables it, the `FAK_USAGE_LOG_PATH` override, what each row contains, and the fact that
nothing rotates the file — lives in
[the durable-artifacts inventory](docs/observability/durable-artifacts.md#the-cli-usage-journal-is-on-by-default).

### Is a slimmer artifact planned?

No. There is no build tag and no reduced release artifact, and the published archives are
the whole binary. That is a design position rather than an oversight: one binary with one
surface is a stated project value — see
[one binary is the whole surface](docs/explainers/one-binary-one-surface.md) — and a second
artifact would be a second thing to build, checksum, attest, and support. If it ever
changes, it will change in a release note rather than silently.

---

## Build from source

If your platform isn't a published target (e.g. 32-bit `linux/arm`), or you want to build
a specific commit, the Go module is the repository root:

```sh
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak        # shipped runtime; needs Go 1.26+
go build -o fak-dev ./cmd/fak-dev # maintainer-only repository tooling
./fak version
```

Adopters install and deploy only `fak`. Maintainers use `fak-dev` directly for repository checks and issue/docs tooling. The legacy `fak dev ...` spelling is a compatibility handoff to a sibling or `PATH`-installed `fak-dev`; it does not link those commands into the runtime.

For a secondary CPU reference build, disable CGo explicitly. Vulkan GPU source
builds require `-tags vulkan` and shader resources; Apple Silicon Metal requires
CGo. The CPU reference cross-compiles cleanly:

```sh
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o fak-linux-arm64 ./cmd/fak
```

---

## About `go install`

`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` installs the latest released
`fak` onto your `$(go env GOBIN)` (`$GOPATH/bin`). The Go module is the repository root, so
the `...@latest` pseudo-path resolves directly — no clone needed. You can equally use the
prebuilt-binary download (sections 1 and 2 above) or the build-from-source fallback.
