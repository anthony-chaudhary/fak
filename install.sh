#!/bin/sh
# install.sh — one-line installer for `fak`, the Fused Agent Kernel.
#
#   curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh
#
# Downloads the prebuilt static binary for your OS/arch from the latest GitHub
# release (attached by .github/workflows/release-artifacts.yml), verifies its
# SHA-256 against the release's SHA256SUMS, and installs it onto your PATH. No
# clone, no Go toolchain; the default cpu variant is pure-Go with no cgo.
# POSIX sh; needs curl (or wget), tar, and sha256sum (or shasum).
#
# Variant (command-line):
#   --variant cpu     pure-Go build, runs everywhere (default)
#   --variant metal   darwin/arm64 (Apple Silicon) only - CGO build with the Metal
#                     backend compiled in, published as
#                     fak_<VERSION>_darwin_arm64_metal.tar.gz
#
# Knobs (environment):
#   FAK_VERSION       pin a version, e.g. 0.24.0 (default: latest release)
#   FAK_INSTALL_DIR   install target (default: /usr/local/bin, else ~/.local/bin)
#   FAK_PREFER_DIRECT bypass Homebrew on macOS and force direct binary download (default: 0)
#   PUBLISH_REPO      owner/repo every published URL derives from — clone,
#                     install, and releases (default: anthony-chaudhary/fak;
#                     FAK_REPO is a back-compat alias)
set -eu

# --- variant selection (--variant cpu|metal) ------------------------------------
VARIANT=cpu
while [ $# -gt 0 ]; do
  case "$1" in
    --variant)
      [ $# -ge 2 ] || { printf 'install.sh: --variant requires an argument (cpu|metal)\n' >&2; exit 2; }
      VARIANT="$2"; shift 2 ;;
    --variant=*) VARIANT="${1#--variant=}"; shift ;;
    -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'install.sh: unknown argument %s — usage: install.sh [--variant cpu|metal]\n' "$1" >&2; exit 2 ;;
  esac
done
case "$VARIANT" in
  cpu|metal) : ;;
  *) printf 'install.sh: unknown variant %s — usage: install.sh [--variant cpu|metal]\n' "$VARIANT" >&2; exit 2 ;;
esac

# Single source of truth for every published URL (clone, install, releases).
PUBLISH_REPO="${PUBLISH_REPO:-${FAK_REPO:-anthony-chaudhary/fak}}"
REPO="$PUBLISH_REPO"
API="https://api.github.com/repos/${REPO}"
DL="https://github.com/${REPO}/releases/download"

err() { printf 'install.sh: %s\n' "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- fetch helper (curl or wget) ------------------------------------------------
fetch() { # fetch URL DEST
  if have curl; then curl -fsSL "$1" -o "$2"
  elif have wget; then wget -qO "$2" "$1"
  else err "need curl or wget"; fi
}
fetch_stdout() { # fetch URL -> stdout
  if have curl; then curl -fsSL "$1"
  elif have wget; then wget -qO- "$1"
  else err "need curl or wget"; fi
}

# --- detect target -------------------------------------------------------------
os="$(uname -s)"; arch="$(uname -m)"
case "$os" in
  Linux)  GOOS=linux ;;
  Darwin) GOOS=darwin ;;
  *) err "unsupported OS '$os' — see INSTALL.md §2 (Manual download), or build from source: go build ./cmd/fak" ;;
esac
case "$arch" in
  x86_64|amd64) GOARCH=amd64 ;;
  arm64|aarch64) GOARCH=arm64 ;;
  *) err "unsupported arch '$arch' — see INSTALL.md §2 (Manual download), or build from source: go build ./cmd/fak" ;;
esac
# Every combination the OS/arch cases above can produce — {linux,darwin}/{amd64,arm64} —
# is now a published .tar.gz target, including linux/arm64 (Raspberry Pi / Jetson / arm64
# gateway). Windows already errored in the OS case above (it ships a .zip, not this tarball).

# The metal variant is a CGO darwin/arm64-only release asset; say so plainly elsewhere.
if [ "$VARIANT" = "metal" ]; then
  if [ "$GOOS" != "darwin" ] || [ "$GOARCH" != "arm64" ]; then
    printf 'install.sh: the metal variant supports darwin/arm64 (Apple Silicon) only; this host is %s/%s — use the default cpu variant or build from source with CGO_ENABLED=1\n' "$GOOS" "$GOARCH" >&2
    exit 2
  fi
fi

# On macOS, prefer Homebrew when available unless an explicit install dir, version pin,
# or direct binary download is requested.
if [ "$GOOS" = "darwin" ] && have brew && [ -z "${FAK_INSTALL_DIR:-}" ] && [ -z "${FAK_VERSION:-}" ] && [ "${FAK_PREFER_DIRECT:-0}" != "1" ]; then
  if brew tap-info anthony-chaudhary/tap >/dev/null 2>&1; then
    printf 'install.sh: macOS detected with Homebrew available — attempting install via Homebrew\n' >&2
    if brew install anthony-chaudhary/tap/fak 2>/dev/null || brew install fak 2>/dev/null; then
      printf 'install.sh: Homebrew install OK\n' >&2
      exit 0
    fi
    printf 'install.sh: Homebrew install failed; proceeding with direct binary download\n' >&2
  else
    printf 'install.sh: skipping Homebrew (tap %s is not installed); using the direct release archive\n' "$PUBLISH_REPO" >&2
  fi
fi

# --- resolve version -----------------------------------------------------------
VERSION="${FAK_VERSION:-}"
if [ -z "$VERSION" ]; then
  # Latest release tag, without pulling in jq.
  VERSION="$(fetch_stdout "${API}/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\{0,1\}\([^"]*\)".*/\1/p' \
    | head -n1)"
  [ -n "$VERSION" ] || err "could not resolve the latest release tag from ${API}/releases/latest"
fi
TAG="v${VERSION}"
if [ "$VARIANT" = "metal" ]; then
  # Metal ships as its own CGO-built asset, separate from the pure-Go cpu archive.
  NAME="fak_${VERSION}_darwin_arm64_metal"
else
  NAME="fak_${VERSION}_${GOOS}_${GOARCH}"
fi
ARCHIVE="${NAME}.tar.gz"

# --- download + verify ---------------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf 'install.sh: downloading %s (%s)\n' "$ARCHIVE" "$TAG" >&2
if ! fetch "${DL}/${TAG}/${ARCHIVE}" "${tmp}/${ARCHIVE}"; then
  if [ "$VARIANT" = "metal" ]; then
    err "the metal asset ${ARCHIVE} was not found for release ${TAG} — install the cpu variant (default) or build from source with CGO_ENABLED=1 for Metal. See ${API}/releases"
  fi
  err "download failed — does release ${TAG} carry a ${GOOS}/${GOARCH} asset? See ${API}/releases"
fi

# Prefer the aggregate SHA256SUMS; fall back to the per-asset .sha256. `sha256sum`
# writes "<hash> *<name>" (a `*` binary-mode marker), so match the name as a field
# with any leading `*` stripped — a `grep " <name>$"` would miss the `*` form.
if fetch "${DL}/${TAG}/SHA256SUMS" "${tmp}/SHA256SUMS" 2>/dev/null; then
  want="$(awk -v a="$ARCHIVE" '{n=$2; sub(/^[*]/,"",n); if (n==a) {print $1; exit}}' "${tmp}/SHA256SUMS")"
else
  fetch "${DL}/${TAG}/${ARCHIVE}.sha256" "${tmp}/${ARCHIVE}.sha256" \
    || err "no checksum published for ${ARCHIVE} — refusing to install unverified"
  want="$(awk '{print $1}' "${tmp}/${ARCHIVE}.sha256" | head -n1)"
fi
[ -n "${want:-}" ] || err "no checksum entry for ${ARCHIVE}"

if have sha256sum; then got="$(sha256sum "${tmp}/${ARCHIVE}" | awk '{print $1}')"
elif have shasum; then got="$(shasum -a 256 "${tmp}/${ARCHIVE}" | awk '{print $1}')"
else err "need sha256sum or shasum to verify the download"; fi
[ "$want" = "$got" ] || err "checksum mismatch for ${ARCHIVE} (want ${want}, got ${got})"
printf 'install.sh: checksum OK\n' >&2

# --- extract + install ---------------------------------------------------------
tar -C "$tmp" -xzf "${tmp}/${ARCHIVE}"
[ -f "${tmp}/fak" ] || err "archive did not contain the fak binary"
chmod +x "${tmp}/fak"

# Pick an install dir we can actually write to, without auto-sudo.
DEST="${FAK_INSTALL_DIR:-}"
if [ -z "$DEST" ]; then
  if [ -w /usr/local/bin ] 2>/dev/null; then DEST=/usr/local/bin
  elif [ "$GOOS" = "darwin" ] && [ -w /opt/homebrew/bin ] 2>/dev/null; then DEST=/opt/homebrew/bin
  else DEST="${HOME}/.local/bin"; fi
fi
mkdir -p "$DEST" || err "cannot create install dir ${DEST}"
[ -w "$DEST" ] || err "no write permission to ${DEST} — re-run with FAK_INSTALL_DIR=~/.local/bin, or sudo"
mv "${tmp}/fak" "${DEST}/fak"

# On macOS, clear quarantine attribute so Gatekeeper allows immediate one-touch execution.
if [ "$GOOS" = "darwin" ] && have xattr; then
  xattr -d com.apple.quarantine "${DEST}/fak" 2>/dev/null || true
fi

printf 'install.sh: installed fak %s to %s/fak\n' "$VERSION" "$DEST" >&2
case ":${PATH}:" in
  *":${DEST}:"*) : ;;
  *) printf 'install.sh: note — %s is not on your PATH; add it or move the binary.\n' "$DEST" >&2 ;;
esac
"${DEST}/fak" version 2>/dev/null || true
