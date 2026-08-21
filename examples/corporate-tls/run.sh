#!/usr/bin/env bash
# corporate-tls: seven witnesses for the managed-host trust seam (#8172) —
# TLS interception and a request-signed cloud upstream, named at launch instead of
# surfacing as a network error or a 24h wait for a login that cannot happen.
#
# Fully OFFLINE and side-effect free: every `fak doctor trust` run here passes
# --probe=false, and every `fak guard` run is stopped by design at a gate (or by an
# --api-key-env that is deliberately unset) BEFORE any gateway binds, any child
# spawns, or any credential is read. No key, no model, no GPU, no network.
set -uo pipefail   # deliberately NOT -e: most witnesses assert on a NON-zero exit

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FAK_BIN="${FAK_BIN:-fak}"
if ! command -v "$FAK_BIN" >/dev/null 2>&1; then
  echo "corporate-tls: fak binary not found; set FAK_BIN=/path/to/fak or put fak on PATH" >&2
  exit 2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OUT="$TMP/out.txt"

pass=0
fail=0
ok()  { printf '  PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad() { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }

# expect_exit WANT GOT LABEL
expect_exit() {
  if [ "$2" = "$1" ]; then ok "exit $2 — $3"; else bad "exit $2, want $1 — $3"; fi
}
# expect_text NEEDLE LABEL  (searches $OUT)
expect_text() {
  if grep -qF -- "$1" "$OUT"; then ok "$2"; else bad "$2 — expected to see: $1"; fi
}
# refute_text NEEDLE LABEL
refute_text() {
  if grep -qF -- "$1" "$OUT"; then bad "$2 — must NOT appear: $1"; else ok "$2"; fi
}

# hostpath renders a path the way the fak BINARY reads it. On Git Bash the shell
# speaks /tmp/... while a native Windows fak.exe needs C:\...; everywhere else this
# is the identity function.
hostpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi
}

# CLEAN is the baseline environment for every witness: the trust seam's own knob,
# the five sibling trust variables it derives, both cloud-route selectors, and the
# waiver — all unset, so each witness declares exactly the one fact it is about and
# the run is identical on a corporate box and a laptop.
CLEAN=(env
  -u FAK_CA_BUNDLE
  -u NODE_EXTRA_CA_CERTS -u AWS_CA_BUNDLE -u CURL_CA_BUNDLE
  -u SSL_CERT_FILE -u SSL_CERT_DIR -u REQUESTS_CA_BUNDLE -u GIT_SSL_CAINFO
  -u CLAUDE_CODE_USE_BEDROCK -u CLAUDE_CODE_USE_VERTEX
  -u FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM
  -u FAK_CORPORATE_TLS_EXAMPLE_KEY)

# GUARD_STOP names an env var that is guaranteed unset. On the Anthropic wire an
# explicitly-named-but-empty --api-key-env is a hard exit(2) (billing must never
# silently fall back), which makes it a reliable full stop AFTER the enterprise
# gates and BEFORE anything binds or spawns. Every guard witness carries it so no
# witness can start a gateway even if its gate unexpectedly passes.
GUARD_STOP=(--api-key-env FAK_CORPORATE_TLS_EXAMPLE_KEY)

echo "== 1/7  a clean host: the check is silent =="
"${CLEAN[@]}" "$FAK_BIN" doctor trust --probe=false >"$OUT" 2>&1
code=$?
cat "$OUT"
expect_exit 0 "$code" "no corporate trust declared, nothing to report"
expect_text "no corporate trust source declared" "names the platform-store posture as correct, not as a gap"
expect_text "doctor: healthy (0 findings)" "zero findings"
refute_text "[WARN" "not one warning on an unintercepted host"
echo

echo "== 2/7  a declared bundle: the trust store fak validates with, named =="
# A throwaway root, minted here rather than committed: a certificate in the repo is
# a fixture that expires, and this one exists for ~2 seconds. Note the paths handed
# to openssl are HOST paths and MSYS_NO_PATHCONV keeps /CN=… from being read as one
# — Git Bash otherwise mangles both halves in opposite directions.
CA="$TMP/corp-root.pem"
mint_root() {
  MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
    -keyout "$(hostpath "$TMP/corp-root.key")" -out "$(hostpath "$CA")" \
    -subj "/CN=Example Corp Root CA/O=Example Corp" "$@" >/dev/null 2>&1
}
if command -v openssl >/dev/null 2>&1 &&
  { mint_root -addext "basicConstraints=critical,CA:TRUE" || mint_root; }; then
  CA_HOST="$(hostpath "$CA")"
  "${CLEAN[@]}" FAK_CA_BUNDLE="$CA_HOST" "$FAK_BIN" doctor trust --probe=false >"$OUT" 2>&1
  code=$?
  cat "$OUT"
  expect_exit 0 "$code" "a bundle that loads is a healthy host, not a finding"
  expect_text "validating against the platform trust store PLUS" "widen, never replace — the platform store is still in the pool"
  expect_text "Example Corp Root CA" "the CA subject the bundle actually carries"
  expect_text "children inherit the same file via NODE_EXTRA_CA_CERTS" "one declaration, derived for every child runtime"
  expect_text "child-runtime-trust" "and it says which child runtimes would NOT inherit it"
else
  echo "  SKIP  witness 2 needs openssl to mint a throwaway root (nothing else here does)"
fi
echo

echo "== 3/7  the no-network signal: a sibling runtime knows something fak does not =="
# The most common real finding, and it needs no handshake: someone already told
# ONE runtime on this host about the interception. fak reading that and staying on
# the platform default IS the "works in my shell, degrades inside fak" bug.
SIB="${CA:-$TMP/corp-root.pem}"
[ -f "$SIB" ] || printf '%s\n' "-----BEGIN CERTIFICATE-----" "(placeholder)" "-----END CERTIFICATE-----" >"$SIB"
"${CLEAN[@]}" AWS_CA_BUNDLE="$(hostpath "$SIB")" "$FAK_BIN" doctor trust --probe=false >"$OUT" 2>&1
code=$?
cat "$OUT"
expect_exit 1 "$code" "a finding exits 1, so CI can route on it"
expect_text "sibling-trust-vars" "the sibling row"
expect_text "already point at a corporate CA bundle on this host" "names the runtime that already knew"
expect_text "export FAK_CA_BUNDLE=" "and hands over the exact same file to declare"
echo

echo "== 4/7  a declared bundle that does not load: UPSTREAM_TRUST_UNVERIFIED =="
# A file that EXISTS and holds no CERTIFICATE block — the ticket text pasted into a
# .pem, or a DER export. The launch refuses rather than falling back to the platform
# store and failing later on a chain nothing in the error names.
NOT_PEM="$TMP/not-a-pem.txt"
echo "Hi, please find the root CA attached." >"$NOT_PEM"
"${CLEAN[@]}" FAK_CA_BUNDLE="$(hostpath "$NOT_PEM")" "$FAK_BIN" guard "${GUARD_STOP[@]}" -- claude >"$OUT" 2>&1
code=$?
cat "$OUT"
expect_exit 2 "$code" "refused before the gateway binds or the child spawns"
expect_text "reason: UPSTREAM_TRUST_UNVERIFIED" "the closed-vocabulary token"
expect_text "next:   fak recover UPSTREAM_TRUST_UNVERIFIED" "bound to its own recovery plan, with the path filled in"
expect_text "check:  fak doctor trust" "and to the read-only command that shows what fak saw"
refute_text "insecure" "no skip-verify escape is ever offered"
echo

echo "== 5/7  a request-signed cloud upstream: UPSTREAM_UNSUPPORTED =="
# This is the bail that replaced a 24h STALE_CRED park: a Bedrock-routed child never
# reads ANTHROPIC_BASE_URL, so fak's gateway would adjudicate nothing — and the
# credential it used to complain about was never the problem.
"${CLEAN[@]}" CLAUDE_CODE_USE_BEDROCK=1 "$FAK_BIN" guard "${GUARD_STOP[@]}" -- claude >"$OUT" 2>&1
code=$?
cat "$OUT"
expect_exit 2 "$code" "refused at the posture, before any credential is read"
expect_text "reason: UPSTREAM_UNSUPPORTED" "the closed-vocabulary token"
expect_text "ignores ANTHROPIC_BASE_URL" "names the mechanism that went inert"
expect_text "works:  fak serve --stdio --policy FILE" "and the route that DOES work on this posture"
refute_text "STALE_CRED" "never a credential refusal for a credential that is fine"
refute_text "setup-token" "and never advice that cannot apply here"
echo

echo "== 6/7  the waiver is loud, not silent =="
"${CLEAN[@]}" CLAUDE_CODE_USE_BEDROCK=1 FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1 \
  "$FAK_BIN" guard "${GUARD_STOP[@]}" -- claude >"$OUT" 2>&1
code=$?
cat "$OUT"
expect_text "proceeding under FAK_GUARD_ALLOW_UNSUPPORTED_UPSTREAM=1" "the waiver announces itself on every launch"
expect_text "fak will see NONE of this session's model traffic" "in the words that matter"
refute_text "reason: UPSTREAM_UNSUPPORTED" "the refusal is downgraded, so the bail block is gone"
expect_text "FAK_CORPORATE_TLS_EXAMPLE_KEY is set but that env var is empty" "the launch then stops on this example's own guardrail — nothing was spawned"
echo

echo "== 7/7  both tokens have a recovery plan =="
"${CLEAN[@]}" "$FAK_BIN" recover UPSTREAM_TRUST_UNVERIFIED >"$OUT" 2>&1
cat "$OUT"
expect_text "fak doctor trust" "UPSTREAM_TRUST_UNVERIFIED routes to the read-only check"
expect_text "never replaces" "and explains widen-never-replace where the operator is standing"
"${CLEAN[@]}" "$FAK_BIN" recover UPSTREAM_UNSUPPORTED >"$OUT" 2>&1
cat "$OUT"
expect_text "fak serve --stdio" "UPSTREAM_UNSUPPORTED routes to the MCP path"
expect_text "not a credential failure" "and says plainly what it is not"
echo

echo "================================================================"
printf 'corporate-tls: %d passed, %d failed\n' "$pass" "$fail"
if [ "$fail" -ne 0 ]; then
  echo "corporate-tls: FAIL"
  exit 1
fi
cat <<'SUMMARY'
corporate-tls: PASS

  Nothing here needed a proxy, a certificate store, admin rights, or -k. A clean
  host raised nothing; a declared bundle was named along with the CA subjects it
  carries and the child runtimes that inherit it; a sibling runtime's own bundle
  was enough to catch the gap with no network at all; and the two postures fak
  cannot silently absorb — a bundle that will not load, and a request-signed cloud
  route — were refused at launch with a token, a check, and a recovery plan.
SUMMARY
