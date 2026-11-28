# Installing `fak`

`fak` runs local models and governs agent tool calls. Public installation defaults
prioritize useful GPU builds: Apple Silicon Metal, Linux amd64 Vulkan (including
AMD Strix Halo), and the NVIDIA CUDA container. CPU builds are secondary references
available through explicit selection. This page is for adopters installing without
cloning the repository.

## Local model setup

| Platform | Install |
| --- | --- |
| Windows x64 | Follow the [PowerShell instructions](#windows-powershell) for the release zip. |
| Apple Silicon Mac | Use the [one-line installer](#1-one-line-installer-recommended) for Metal. |
| Linux x64 | Use the [CPU installer](#1-one-line-installer-recommended) with `--variant cpu` for the reference build. The default Vulkan build requires the driver and glibc versions listed below. |

Once `fak version` works, [start the coding model](README.md#try-fak). See
[local model setup](GETTING-STARTED.md#local-model-setup) for platform verification.

## Other install options

Three supported paths, fastest first:

1. [One-line installer](#1-one-line-installer-recommended) — `curl | sh`, downloads a
   verified prebuilt binary.
2. [Manual download](#2-manual-download) — grab the archive for your OS/arch from the
   GitHub release yourself.
3. [Docker](#3-docker) — the NVIDIA CUDA image, or a CPU reference image.

[What's in the binary](#whats-in-the-binary) answers the question the three paths above
skip — why one ~13 MB binary carries hundreds of verbs, and which of them are yours. A
[build-from-source](#build-from-source) fallback and the [`go install`
status](#about-go-install) are at the end. The full map of options — Homebrew,
Docker/OCI, Compose, Kubernetes/Helm, systemd, winget, macOS launchd — is in the
[install-options matrix](#install-options-at-a-glance).

Once `fak` is on your PATH, [start your first model](README.md#try-fak).

The GPU archive contract is `darwin/arm64` Metal and `linux/amd64` Vulkan.
Vulkan archives require GNU glibc 2.39+ (Ubuntu 24.04+), the system Vulkan
loader (`libvulkan.so.1`), and a compatible GPU driver. They include a `spirv/`
directory that must stay beside the binary. The installer rejects older glibc;
build Vulkan from source for older distributions.
CPU reference archives remain available for `linux/amd64`, `linux/arm64`,
`darwin/amd64`, `darwin/arm64`, and `windows/amd64`.

The [v0.60.0 release](https://github.com/anthony-chaudhary/fak/releases/tag/v0.60.0)
includes the Apple Silicon Metal and Linux amd64 Vulkan archives, a checksum
alongside each download, and the aggregate `SHA256SUMS`.
A packaged backend still needs qualification on the actual model and GPU.

---

## Install options at a glance

The three paths above plus every packaging option in one place. Running the
gateway in production (auth, policy, sizing, service units) is
[`docs/fak/deployment-guide.md`](docs/fak/deployment-guide.md).

| Option | How | Best for |
| --- | --- | --- |
| One-line installer | [`curl \| sh`](#1-one-line-installer-recommended) | Linux/macOS laptops and servers; verifies against `SHA256SUMS` |
| Manual download | [release archive](#2-manual-download) | Air-gapped hosts; the Windows `.zip` |
| `go install` | [`go install …@latest`](#about-go-install) | Go developers |
| Homebrew | [`Formula/fak.rb`](Formula/fak.rb) | macOS (and Linuxbrew); installs the CPU build |
| Docker / OCI | [`ghcr.io/anthony-chaudhary/fak`](#3-docker) | Containers; published multi-arch `linux/amd64` + `linux/arm64` (plus the NVIDIA CUDA image) |
| Docker Compose | [`examples/compose-ollama`](examples/compose-ollama) | fak + Ollama on one box |
| Kubernetes / Helm | [`deploy/k8s`](deploy/k8s), [`deploy/helm`](deploy/helm) | Clusters — `kubectl apply -k deploy/k8s`; Helm from a checkout: `helm install fak deploy/helm/fak`; Helm without a clone: `helm install fak oci://ghcr.io/anthony-chaudhary/fak/charts/fak --version X.Y.Z` (Helm 3.8+) |
| systemd | [`deploy/systemd`](deploy/systemd) | Linux service — hardened unit, `DynamicUser` |
| winget | [`deploy/winget`](deploy/winget) | Windows — `winget install AnthonyChaudhary.Fak` (after the one-time submission) |
| macOS launchd | [`tools/com.fak.serve-gateway.plist`](tools/com.fak.serve-gateway.plist) | macOS service |

---

## 1. One-line installer (recommended)

```sh
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

The installer ([`install.sh`](install.sh)) defaults to Metal on Apple Silicon and
Vulkan on Linux amd64. If `nvidia-smi -L` detects NVIDIA and a working `docker`
is available, it prints the CUDA container command. Without docker it builds the
CUDA variant from source on the host (`--variant cuda`, see
[NVIDIA CUDA from source](#nvidia-cuda-from-source)). Other platforms require
explicit `--variant cpu` for a CPU reference. Missing GPU archives never silently
fall back to CPU.

Downloads are verified against `SHA256SUMS` before installation. Vulkan shader
resources are installed to `spirv/` beside `fak`. The installer needs POSIX `sh`,
`curl` (or `wget`), `tar`, and `sha256sum` (or `shasum`).

```sh
# Explicit CPU reference:
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh -s -- --variant cpu
```

Then confirm:

```sh
fak version          # prints the installed version, e.g. 0.60.0
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
| `FAK_VERSION` | Pin a published release version | latest release |
| `FAK_INSTALL_DIR` | Install target directory | `/usr/local/bin` if writable, else `~/.local/bin` |
| `FAK_REPO` | `owner/repo` override | `anthony-chaudhary/fak` |

Install the latest Metal build on Apple Silicon in your user-local directory.

```sh
FAK_INSTALL_DIR="$HOME/.local/bin" \
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
VERSION=0.59.0
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
$Release = Invoke-RestMethod 'https://api.github.com/repos/anthony-chaudhary/fak/releases/latest'
$Version = $Release.tag_name.TrimStart('v')
$Archive = "fak_${Version}_windows_amd64.zip"
$Base    = "https://github.com/anthony-chaudhary/fak/releases/download/v$Version"

Invoke-WebRequest "$Base/$Archive" -OutFile $Archive
Invoke-WebRequest "$Base/SHA256SUMS" -OutFile SHA256SUMS

# Verify
$want = (Select-String " $Archive$" SHA256SUMS).Line.Split()[0]
$got  = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower()
if ($want -ne $got) { throw "checksum mismatch for $Archive" }

Expand-Archive $Archive -DestinationPath fak-dist
# Verify the extracted binary before adding its folder to PATH.
.\fak-dist\fak.exe version
```

Keep the extracted folder somewhere permanent. In the Start menu, search for
**Edit environment variables for your account**. Under **User variables**, edit
**Path**, choose **New**, and add the full path to the folder containing `fak.exe`.
Open a new terminal and run `fak version`.

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
`Dockerfile.cuda` compiles `sm_120f` for Blackwell consumer GPUs (RTX 50-series)
so native NVFP4 W4A4 prefill is active out of the box rather than falling back to
int8. To build a single-arch local image for an RTX 50-series card:

```sh
docker build -f Dockerfile.cuda --build-arg CUDA_ARCH=sm_120f -t fak:cuda .
```

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
docker build --build-arg APP_VERSION=0.60.0 -t fak:0.60.0 .
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
rows. The classification is enforced by tests so a verb cannot drift into the wrong column.

### Is the dev tier a problem in a production image?

No — and the reason is specific, not reassurance:

- **A verb runs only when it is the command you invoke.** Nothing in the dev tier is
  scheduled or started by `fak serve`, and no HTTP or MCP request can reach one: the
  gateway never spawns a process.
- **The container never invokes one.** The [`Dockerfile`](Dockerfile) sets
  `ENTRYPOINT ["/usr/local/bin/fak"]` with `CMD ["serve", "--addr", "0.0.0.0:8080"]`.
- **Most dev verbs have nothing to act on outside a checkout.** They locate the repository
  by searching upward for `go.mod`; a distroless image has none.
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

### NVIDIA CUDA from source

No CUDA release archive is published: the CUDA binary links the host's NVIDIA
driver (`libcuda.so.1`) and CUDA toolkit runtime. Without docker, build it on the
GPU host. The installer does this for you:

```sh
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh -s -- --variant cuda
```

It needs `git`, Go 1.26+, `g++`, the NVIDIA driver (`nvidia-smi`) and the CUDA
toolkit (`nvcc`). It finds the toolkit from `CUDA_HOME`, the `nvcc` on `PATH`,
`/usr/local/cuda` or `/opt/cuda` (Arch), reads the GPU arch from `nvidia-smi`
— automatically selecting `sm_120f` on Blackwell consumer / SM12x cards (CC
12.0/12.1, RTX 50-series) so native NVFP4 W4A4 prefill compiles in instead of
silently falling back to int8 (requires CUDA 12.9+; override with
`FAK_CUDA_ARCH=sm_120a` on CUDA 12.8, or `FAK_CUDA_ARCH=sm_120f` explicitly) —
keeps the checkout in `~/.cache/fak/src` (`FAK_SRC_DIR`), builds `main`
(`FAK_REF`, or `FAK_VERSION` for a release tag), and embeds the toolkit library
dir as the binary's rpath so `fak` runs without `LD_LIBRARY_PATH`. The manual
equivalent on an RTX 50-series host with `CUDA_HOME=/opt/cuda` (or
`make cuda-build-sm120f` inside the checkout):

```sh
git clone --depth 1 https://github.com/anthony-chaudhary/fak.git && cd fak
lib="$CUDA_HOME/targets/x86_64-linux/lib"   # or $CUDA_HOME/lib64
export CGO_ENABLED=1 LIBRARY_PATH="$lib" CGO_LDFLAGS="-L$lib -Wl,-rpath,$lib"
go run ./cmd/fak-dev build-cuda build --arch sm_120f --cuda-home "$CUDA_HOME"
go build -tags cuda -o ~/.local/bin/fak ./cmd/fak
fak backends   # lists cuda when the driver sees the GPU
```

---

## About `go install`

`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` installs the latest released
`fak` onto your `$(go env GOBIN)` (`$GOPATH/bin`). The Go module is the repository root, so
the `...@latest` pseudo-path resolves directly — no clone needed. You can equally use the
prebuilt-binary download (sections 1 and 2 above) or the build-from-source fallback.

The Go module proxy and GitHub's Latest release both serve v0.59.0, which
includes Inkling-2-Mini. Use `@v0.60.0` with Go or set `FAK_VERSION=0.60.0`
with the prebuilt installer to pin that version. For CUDA, use the
`ghcr.io/anthony-chaudhary/fak:0.59.0-cuda` image or follow the
[source build instructions](#build-from-source).

---

## Uninstall

Run these steps in order; skip any that do not apply to your install.

1. Remove the `claude`/`codex` launch shims, if you created them with `fak launch install`:
   ```bash
   fak launch uninstall --provider all   # or --provider claude | codex
   ```
2. Remove the gateway service, if you installed one with `fak node install`
   (launchd agent, systemd `--user` unit, or Windows Scheduled Task):
   ```bash
   fak node install --uninstall
   ```
3. Remove the binary, matching how you installed it:
   - One-line installer: `rm "$(command -v fak)"`. It installs to `$FAK_INSTALL_DIR`, else
     `/usr/local/bin`, `/opt/homebrew/bin` (macOS), or `~/.local/bin`. A Vulkan install also
     placed a `spirv/` directory beside the binary; remove it first with
     `rm -r "$(dirname "$(command -v fak)")/spirv"`, then remove the binary.
   - Homebrew: `brew uninstall fak`.
   - `go install`: delete `fak` (`fak.exe` on Windows) from `$(go env GOBIN)`, or from
     `$(go env GOPATH)/bin` if GOBIN is unset.
   - Windows zip: delete `fak.exe` from wherever you moved it.
4. Delete downloaded models: remove the whole `$FAK_MODELS_DIR` directory, or by default
   `fak-models` inside your user cache directory (`~/.cache` on Linux, `~/Library/Caches`
   on macOS, `%LocalAppData%` on Windows). It holds plain files (`hub/`, `gguf/`) plus the
   model alias registry.
5. Delete fak's settings and local state: the `~/.fak/` directory, and the `fak` directory
   inside your user config directory (`~/.config` on Linux, `~/Library/Application Support`
   on macOS, `%AppData%` on Windows). If you ran `fak codex config --write`, also remove the
   `fak` provider block it added to that workspace's `.codex/config.toml`.
