# Installing `fak`

`fak` is the Fused Agent Kernel — one static Go binary you
put in front of your model so every tool call is adjudicated before it runs. It has
**zero external dependencies** (standard library only — there is no `go.sum`, no Python,
no CUDA toolchain), so "install" really is just *get the binary onto the box*. This page
is for an **external adopter**: you want `fak` on your machine or in your production
image **without cloning the monorepo**.

Three supported paths, fastest first:

1. [One-line installer](#1-one-line-installer-recommended) — `curl | sh`, downloads a
   verified prebuilt binary.
2. [Manual download](#2-manual-download) — grab the archive for your OS/arch from the
   GitHub release yourself.
3. [Docker](#3-docker) — a tiny distroless image for "put `fak` in front of my model in
   production".

A [build-from-source](#build-from-source) fallback and the [`go install`
status](#about-go-install) are at the end.

Next: once `fak` is on your PATH, the [first-session tutorial](docs/fak/tutorial.md)
walks you from the binary to your first adjudicated tool call — fully offline, no key or GPU.

The published targets are **`linux/amd64`, `darwin/amd64`, `darwin/arm64`,
`windows/amd64`**. Each `vX.Y.Z` release attaches one archive per target plus a
`SHA256SUMS` file; the release workflow that produces them is
[`.github/workflows/release-artifacts.yml`](.github/workflows/release-artifacts.yml).

---

## 1. One-line installer (recommended)

```sh
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
```

The installer ([`install.sh`](install.sh)) detects your OS/arch, downloads the prebuilt
static binary for the **latest** release, **verifies its SHA-256 against the release's
`SHA256SUMS`** (it refuses to install an unverified download), and drops `fak` onto your
PATH. No clone, no Go toolchain, no cgo. It needs only POSIX `sh`, `curl` (or `wget`),
`tar`, and `sha256sum` (or `shasum`).

Then confirm:

```sh
fak version          # prints the installed version, e.g. 0.33.0
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
| `FAK_VERSION` | Pin a version, e.g. `0.33.0` | latest release |
| `FAK_INSTALL_DIR` | Install target directory | `/usr/local/bin` if writable, else `~/.local/bin` |
| `FAK_REPO` | `owner/repo` override | `anthony-chaudhary/fak` |

Example — pin a version into a user-local dir:

```sh
FAK_VERSION=0.33.0 FAK_INSTALL_DIR="$HOME/.local/bin" \
  sh -c 'curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh'
```

> Piping a script into a shell runs it with your privileges. If you'd rather read it
> first, download `install.sh`, inspect it, then run `sh install.sh`.

---

## 2. Manual download

If you don't want to run an installer, take the archive straight from the
[Releases page](https://github.com/anthony-chaudhary/fak/releases). Assets are named:

```
fak_<version>_<os>_<arch>.tar.gz     # linux/darwin
fak_<version>_<os>_<arch>.zip        # windows
fak_<version>_<os>_<arch>.tar.gz.sha256
SHA256SUMS                           # aggregate, all targets
```

### Linux / macOS

```sh
VERSION=0.33.0
OS=$(uname -s | tr '[:upper:]' '[:lower:]')          # linux | darwin
ARCH=$(uname -m); [ "$ARCH" = x86_64 ] && ARCH=amd64; [ "$ARCH" = aarch64 ] && ARCH=arm64
ARCHIVE="fak_${VERSION}_${OS}_${ARCH}.tar.gz"
BASE="https://github.com/anthony-chaudhary/fak/releases/download/v${VERSION}"

curl -fsSLO "${BASE}/${ARCHIVE}"
curl -fsSLO "${BASE}/SHA256SUMS"

# Verify before trusting the binary.
grep " ${ARCHIVE}\$" SHA256SUMS | sha256sum -c -

tar -xzf "${ARCHIVE}"        # extracts: fak, LICENSE, GETTING-STARTED.md
chmod +x fak
sudo mv fak /usr/local/bin/  # or any dir on your PATH
fak version
```

### Windows (PowerShell)

```powershell
$Version = "0.33.0"
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

For "put `fak` in front of my model in production", build the image from the
[`Dockerfile`](Dockerfile) at the repo root. It's a two-stage build: the pure-Go binary
is compiled static (`CGO_ENABLED=0`) and copied into a `distroless/static` base, so the
final image is just that base plus one ~13 MB binary — no shell, no package manager, no
libc, running as nonroot. (That's the *governance surface*. A GPU token engine like vLLM
or SGLang ships a multi-GB image — roughly 8–12 GB compressed in current tags — because it
bundles CUDA and PyTorch by design. `fak` *fronts* that engine rather than containing it,
so its own image stays tiny and cold-starts instantly.)

```sh
docker build -t fak .

# Front a model served by Ollama on the host:
docker run --rm -p 8080:8080 fak serve --addr 0.0.0.0:8080 \
    --base-url http://host.docker.internal:11434/v1 --model qwen2.5:1.5b
```

Containers must bind `0.0.0.0`, not loopback — the default `CMD` already does
(`serve --addr 0.0.0.0:8080`). Stamp a specific version into the binary at build time:

```sh
docker build --build-arg APP_VERSION=0.33.0 -t fak:0.33.0 .
```

Override the entrypoint command to run `fak agent`, `fak policy`, etc. instead of the
gateway.

---

## Build from source

If your platform isn't a published target (e.g. `linux/arm64`), or you want to build a
specific commit, the Go module is the repository root:

```sh
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak        # needs Go 1.26+
./fak version
```

The default binary is pure Go with no cgo (the Vulkan compute backend is behind a build
tag and absent from the default build), so it cross-compiles cleanly:

```sh
GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build -o fak-linux-arm64 ./cmd/fak
```

---

## About `go install`

`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` installs the latest released
`fak` onto your `$(go env GOBIN)` (`$GOPATH/bin`). The Go module is the repository root, so
the `...@latest` pseudo-path resolves directly — no clone needed. You can equally use the
prebuilt-binary download (sections 1 and 2 above) or the build-from-source fallback.
