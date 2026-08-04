// runmanifest.go — the replay-complete quality-run provenance manifest
// (#4514, epic #4509: the missing middle validation ladder).
//
// The trust half of this package (provenance.go) answers "where did this BYTE
// come from, and may the kernel trust it?" for a single tool-call result. This
// half answers the sibling question for a whole QUALITY RUN: "under exactly what
// binary revision, model, tokenizer, hardware, backend, decode config, cache
// state, output normalization, and environment was this eval produced — and can
// an independent replay reproduce it identically?"
//
// It exists because leading eval stacks (OpenAI Evals, HELM) version the task,
// the run metadata, AND the artifact rather than treating free-form output as a
// test result. A RunManifest is that versioned record: the independently
// verifiable middle layer between fak primitive-correctness tests and coarse
// end-benchmark scores. The acceptance contract maps one-to-one onto the API:
//
//   - "equivalent runs normalize identically / one changed flag is visible"
//     -> Normalize + Fingerprint + Equivalent (same fingerprint) and
//     FirstDivergence (the changed flag, localized and readable).
//   - "each case records model, tokenizer, engine/backend, seed or deterministic
//     oracle, code/module revision, tolerance/baseline provenance" -> the named
//     RunManifest fields, all required by Validate.
//   - "failure identifies the first actionable divergence and emits a scrubbed
//     replay artifact; missing or inconclusive evidence is never pass" ->
//     Compare returns a fail-closed ReplayArtifact whose secrets are redacted.
//   - "assign the case to an explicit PR/nightly/release tier and document
//     runtime/resource cost" -> Tier + Cost, both required by Validate.
//
// Everything here is a pure, stdlib-only library (matching the trust half's
// canon/grammar shape): it decides identity and divergence from recorded facts,
// it never runs a model. Determinism is load-bearing — Go's encoding/json emits
// struct fields in declaration order and sorts map keys, so a normalized manifest
// serializes to stable bytes, which is what makes the fingerprint and the replay
// round-trip reproducible in a clean, independent environment.
package provenance

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Tier is the gate lane a quality case is assigned to (acceptance criterion 4:
// "assign the case to an explicit PR, nightly, or release tier"). The zero value
// is TierUnset — an unassigned case is inconclusive, never a pass.
type Tier uint8

const (
	TierUnset Tier = iota
	TierPR
	TierNightly
	TierRelease
)

// String renders the tier as its lowercase gate name, or "unset" for the
// fail-closed zero value.
func (t Tier) String() string {
	switch t {
	case TierPR:
		return "pr"
	case TierNightly:
		return "nightly"
	case TierRelease:
		return "release"
	}
	return "unset"
}

// Valid reports whether the tier is an explicit assignment (not the fail-closed
// zero value).
func (t Tier) Valid() bool { return t == TierPR || t == TierNightly || t == TierRelease }

// RunManifest is the replay-complete provenance record for one quality case run.
// Every field the acceptance contract names is explicit; the two open sets
// (DecodeParams, Env) are maps so a backend can record its own knobs without a
// schema change, and Normalize canonicalizes them (trim, drop empties, and — via
// canonical marshaling — erase key order) so equivalent runs still fingerprint
// identically while any genuine flag change survives and stays visible.
type RunManifest struct {
	// Case identity — the versioned task id these results belong to.
	Case string `json:"case"`

	// Acceptance criterion 2: model, tokenizer, engine/backend, seed OR
	// deterministic oracle, code/module revision, tolerance/baseline provenance.
	Model     string `json:"model"`     // model id + weight hash, e.g. "glm-4.6@sha256:abcd"
	Tokenizer string `json:"tokenizer"` // tokenizer id + hash

	// LoadProvenance is the model-load provenance digest (#4746, root incident
	// #4273): the content address of the model.LoadProvenance artifact — model
	// bytes, GGUF architecture/version, canonical manifest, loader revision, and
	// every non-identity loader transform — that the run's in-memory tensors were
	// produced under.
	//
	// It is recorded here because Model pins WHICH bytes were loaded and says
	// nothing about what the loader DID to them, and the #4273 defect class is
	// invisible in every other field this manifest records: the broken load and
	// the fixed load agree on model id, tokenizer, backend, hardware, quant, and
	// decode params, and differ only in loader SEMANTICS. Without this field two
	// such runs fingerprint identically and Compare returns pass — a publication
	// claim bound to nothing about the loader.
	//
	// The digest is produced by (*ggufload.File).LoadProvenance(scope).Digest()
	// and is publish-safe by construction: the artifact it addresses cannot hold
	// a prompt, a filesystem path, or a raw weight. Recording the DIGEST rather
	// than the artifact also keeps this package stdlib-only — provenance never
	// imports the loader, it just carries its content address.
	LoadProvenance string `json:"load_provenance"`
	Backend        string `json:"backend"`   // engine / backend, e.g. "fak-engine" vs "llama.cpp"
	Seed           int64  `json:"seed"`      // decode seed; 0 with a non-empty Oracle == deterministic-oracle mode
	Oracle         string `json:"oracle"`    // deterministic oracle id (an alternative to a seed)
	CodeRev        string `json:"code_rev"`  // code/module revision, prefer module@rev over a bare SHA
	Baseline       string `json:"baseline"`  // baseline provenance the tolerance is measured against
	Tolerance      string `json:"tolerance"` // tolerance the comparison used, e.g. "rel<=1e-3"

	// Scope: binary revision, hardware, decode parameters, cache state,
	// output normalization, and environment.
	BinaryRev     string            `json:"binary_rev"`    // fak binary revision the run used
	Hardware      string            `json:"hardware"`      // device/arch, e.g. "A100-80GB" / "cpu-avx512"
	DecodeParams  map[string]string `json:"decode_params"` // temperature, top_p, max_tokens, ...
	CacheState    string            `json:"cache_state"`   // "cold" / "warm-prefix" / "l3-shared" ...
	Normalization string            `json:"normalization"` // output normalization applied, e.g. "nfc+trim"
	Env           map[string]string `json:"env"`           // environment the run observed

	// Acceptance criterion 4: tier assignment + runtime/resource cost.
	Tier Tier   `json:"tier"`
	Cost string `json:"cost"` // runtime/resource cost, e.g. "3.2s / 1xA100 / 4.1k tok"

	// Secrets names extra env/decode keys whose VALUES must be redacted from any
	// emitted replay artifact, on top of the built-in secret-shaped-key heuristic.
	// It never participates in identity (json:"-", excluded from the fingerprint).
	Secrets []string `json:"-"`
}

const redacted = "***"

// Normalize returns a canonicalized copy: every string field is space-trimmed and
// both open maps have their keys/values trimmed with empty-valued entries dropped.
// Two runs that differ only in insignificant whitespace or map ORDER normalize to
// byte-identical manifests (map order is erased by canonical marshaling), so they
// share a Fingerprint — while any genuine field change survives.
func (m RunManifest) Normalize() RunManifest {
	n := m
	n.Case = strings.TrimSpace(m.Case)
	n.Model = strings.TrimSpace(m.Model)
	n.Tokenizer = strings.TrimSpace(m.Tokenizer)
	n.LoadProvenance = strings.TrimSpace(m.LoadProvenance)
	n.Backend = strings.TrimSpace(m.Backend)
	n.Oracle = strings.TrimSpace(m.Oracle)
	n.CodeRev = strings.TrimSpace(m.CodeRev)
	n.Baseline = strings.TrimSpace(m.Baseline)
	n.Tolerance = strings.TrimSpace(m.Tolerance)
	n.BinaryRev = strings.TrimSpace(m.BinaryRev)
	n.Hardware = strings.TrimSpace(m.Hardware)
	n.CacheState = strings.TrimSpace(m.CacheState)
	n.Normalization = strings.TrimSpace(m.Normalization)
	n.Cost = strings.TrimSpace(m.Cost)
	n.DecodeParams = normMap(m.DecodeParams)
	n.Env = normMap(m.Env)
	if len(m.Secrets) > 0 {
		n.Secrets = append([]string(nil), m.Secrets...)
		sort.Strings(n.Secrets)
	} else {
		n.Secrets = nil
	}
	return n
}

func normMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		k = strings.TrimSpace(k)
		v = strings.TrimSpace(v)
		if k == "" || v == "" {
			continue
		}
		out[k] = v
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// scrub returns a normalized copy whose secret-shaped env/decode VALUES are
// replaced with a fixed redaction token. A value is scrubbed when its key matches
// the built-in secret heuristic (contains token/secret/password/api-key/...) OR
// the key is named in Secrets. This is what makes a replay artifact safe to attach
// to a PUBLIC failure bundle — the contract's "scrubbed replay artifact" — and it
// is idempotent, so re-scrubbing a replayed artifact is a no-op.
func (m RunManifest) scrub() RunManifest {
	n := m.Normalize()
	extra := make(map[string]bool, len(n.Secrets))
	for _, k := range n.Secrets {
		extra[strings.ToLower(strings.TrimSpace(k))] = true
	}
	n.DecodeParams = scrubMap(n.DecodeParams, extra)
	n.Env = scrubMap(n.Env, extra)
	return n
}

func scrubMap(in map[string]string, extra map[string]bool) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		if secretShaped(k) || extra[strings.ToLower(k)] {
			out[k] = redacted
		} else {
			out[k] = v
		}
	}
	return out
}

func secretShaped(key string) bool {
	k := strings.ToLower(key)
	for _, needle := range []string{"secret", "password", "passwd", "apikey", "api_key", "_key", "credential"} {
		if strings.Contains(k, needle) {
			return true
		}
	}
	// A credential "token" is secret-shaped, but the PLURAL "tokens" is a decode
	// count (max_tokens, num_tokens) — never a secret. Guarding on the plural keeps
	// the safety-net heuristic from redacting a legitimate decode parameter.
	if strings.Contains(k, "token") && !strings.Contains(k, "tokens") {
		return true
	}
	return false
}

// canonicalBytes serializes the SCRUBBED, normalized manifest to deterministic
// JSON. Secrets never affect identity (they are redacted first), so two runs that
// differ only in a rotated key fingerprint identically — and the scrubbed replay
// artifact reproduces the exact same fingerprint.
func (m RunManifest) canonicalBytes() []byte {
	b, _ := json.Marshal(m.scrub())
	return b
}

// Fingerprint is the content address of the normalized, scrubbed manifest: sha256
// hex over canonicalBytes. Equivalent runs share it; changing any recorded
// (non-secret) field changes it.
func (m RunManifest) Fingerprint() string {
	sum := sha256.Sum256(m.canonicalBytes())
	return hex.EncodeToString(sum[:])
}

// Equivalent reports whether two runs normalize to the identical record — the
// acceptance property "equivalent runs normalize identically".
func (m RunManifest) Equivalent(other RunManifest) bool {
	return m.Fingerprint() == other.Fingerprint()
}

// Validate enforces the "missing or inconclusive evidence is never pass" rule:
// every field the contract requires must be present, the case must carry an
// explicit tier and a documented cost, and the run must pin determinism with
// EITHER a non-zero seed OR a named oracle. A returned error means the manifest is
// inconclusive and can never back a pass.
func (m RunManifest) Validate() error {
	n := m.Normalize()
	var missing []string
	for _, r := range []struct{ name, val string }{
		{"case", n.Case},
		{"model", n.Model},
		{"load_provenance", n.LoadProvenance},
		{"tokenizer", n.Tokenizer},
		{"backend", n.Backend},
		{"code_rev", n.CodeRev},
		{"binary_rev", n.BinaryRev},
		{"hardware", n.Hardware},
		{"baseline", n.Baseline},
		{"tolerance", n.Tolerance},
		{"cache_state", n.CacheState},
		{"normalization", n.Normalization},
		{"cost", n.Cost},
	} {
		if r.val == "" {
			missing = append(missing, r.name)
		}
	}
	if n.Seed == 0 && n.Oracle == "" {
		missing = append(missing, "seed|oracle")
	}
	if !n.Tier.Valid() {
		missing = append(missing, "tier")
	}
	if len(n.DecodeParams) == 0 {
		missing = append(missing, "decode_params")
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("inconclusive manifest: missing %s", strings.Join(missing, ", "))
	}
	// A present-but-unshaped load_provenance is worse than an absent one: it
	// satisfies the presence check while addressing no artifact, so the manifest
	// LOOKS bound to its loader semantics and proves nothing. Check the shape.
	if !sha256Digest(n.LoadProvenance) {
		return fmt.Errorf("inconclusive manifest: load_provenance %q is not a sha256:<hex> content address", n.LoadProvenance)
	}
	return nil
}

// sha256Digest reports whether s is a "sha256:<64 lowercase hex>" content
// address — the form model.LoadProvenance.Digest emits.
func sha256Digest(s string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(s, prefix) {
		return false
	}
	hex := s[len(prefix):]
	if len(hex) != 64 {
		return false
	}
	for i := 0; i < len(hex); i++ {
		if c := hex[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// Divergence is the first field at which a candidate run departs from its baseline
// — the "first actionable divergence" a failure must localize. Field is the
// canonical dotted path (e.g. "decode_params.temperature"); Baseline/Candidate are
// the SCRUBBED readings, safe to publish (a missing side reads as "").
type Divergence struct {
	Field     string `json:"field"`
	Baseline  string `json:"baseline"`
	Candidate string `json:"candidate"`
}

// fields returns the manifest's canonical (path, value) pairs in a FIXED order —
// scalars first in contract order, then decode_params.* and env.* by sorted key.
// Values are scrubbed so a Divergence built from them is publish-safe. The set of
// paths is exactly the set of fingerprinted fields, so two manifests share a
// Fingerprint iff FirstDivergence between them is nil.
func (m RunManifest) fields() [][2]string {
	n := m.scrub()
	pairs := [][2]string{
		{"case", n.Case},
		{"model", n.Model},
		// load_provenance sits directly after model, ahead of every downstream
		// flag: a loader-semantics difference invalidates the comparison of
		// everything below it, so it must be the first divergence an operator
		// reads rather than one they reach after chasing a decode param.
		{"load_provenance", n.LoadProvenance},
		{"tokenizer", n.Tokenizer},
		{"backend", n.Backend},
		{"seed", fmt.Sprintf("%d", n.Seed)},
		{"oracle", n.Oracle},
		{"code_rev", n.CodeRev},
		{"baseline", n.Baseline},
		{"tolerance", n.Tolerance},
		{"binary_rev", n.BinaryRev},
		{"hardware", n.Hardware},
		{"cache_state", n.CacheState},
		{"normalization", n.Normalization},
		{"tier", n.Tier.String()},
		{"cost", n.Cost},
	}
	pairs = append(pairs, mapFields("decode_params", n.DecodeParams)...)
	pairs = append(pairs, mapFields("env", n.Env)...)
	return pairs
}

func mapFields(prefix string, in map[string]string) [][2]string {
	if len(in) == 0 {
		return nil
	}
	keys := make([]string, 0, len(in))
	for k := range in {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([][2]string, 0, len(in))
	for _, k := range keys {
		out = append(out, [2]string{prefix + "." + k, in[k]})
	}
	return out
}

// FirstDivergence returns the first field, in canonical order, where cand departs
// from the baseline m — or nil if the runs are equivalent. A path present on only
// one side diverges against an empty value on the other, so a dropped or added
// flag is caught and visible.
func (m RunManifest) FirstDivergence(cand RunManifest) *Divergence {
	bf := m.fields()
	cf := cand.fields()
	bm := pairsToMap(bf)
	cm := pairsToMap(cf)
	for _, path := range orderedUnion(bf, cf) {
		if bm[path] != cm[path] {
			return &Divergence{Field: path, Baseline: bm[path], Candidate: cm[path]}
		}
	}
	return nil
}

func pairsToMap(pairs [][2]string) map[string]string {
	out := make(map[string]string, len(pairs))
	for _, p := range pairs {
		out[p[0]] = p[1]
	}
	return out
}

func orderedUnion(a, b [][2]string) []string {
	seen := make(map[string]bool, len(a)+len(b))
	order := make([]string, 0, len(a)+len(b))
	for _, src := range [][][2]string{a, b} {
		for _, p := range src {
			if !seen[p[0]] {
				seen[p[0]] = true
				order = append(order, p[0])
			}
		}
	}
	return order
}

// ReplayArtifact is the scrubbed, replay-complete failure bundle: the normalized
// manifest (secrets redacted), its Fingerprint, the verdict, and — on a failure —
// the first divergence from the baseline. It round-trips: ReplayFrom(a.JSON())
// rebuilds a manifest whose Fingerprint equals a.Fingerprint, which is what lets a
// clean, independent environment REPLAY the run identically.
type ReplayArtifact struct {
	Manifest    RunManifest `json:"manifest"`
	Fingerprint string      `json:"fingerprint"`
	Verdict     string      `json:"verdict"` // "pass" | "fail" | "inconclusive"
	Reason      string      `json:"reason,omitempty"`
	Divergence  *Divergence `json:"divergence,omitempty"`
}

// Verdict tokens. "pass" is the ONLY passing value; "fail" and "inconclusive" are
// both non-pass, so missing/inconclusive evidence can never read as a pass.
const (
	verdictPass         = "pass"
	verdictFail         = "fail"
	verdictInconclusive = "inconclusive"
)

// Compare adjudicates a candidate run against its baseline, fail-closed. If either
// manifest is inconclusive (Validate fails) the verdict is NEVER pass. Otherwise
// the runs pass iff they are Equivalent; any divergence fails and is localized to
// its first field. The returned artifact is scrubbed and ready to attach.
func Compare(baseline, candidate RunManifest) ReplayArtifact {
	art := ReplayArtifact{
		Manifest:    candidate.scrub(),
		Fingerprint: candidate.Fingerprint(),
	}
	if err := baseline.Validate(); err != nil {
		art.Verdict, art.Reason = verdictInconclusive, "baseline "+err.Error()
		return art
	}
	if err := candidate.Validate(); err != nil {
		art.Verdict, art.Reason = verdictInconclusive, "candidate "+err.Error()
		return art
	}
	if d := baseline.FirstDivergence(candidate); d != nil {
		art.Verdict = verdictFail
		art.Reason = "run diverged from baseline at " + d.Field
		art.Divergence = d
		return art
	}
	art.Verdict = verdictPass
	return art
}

// Pass is the single boolean gate: only an explicit pass verdict passes.
func (a ReplayArtifact) Pass() bool { return a.Verdict == verdictPass }

// JSON renders the artifact as indented, deterministic JSON — the on-disk replay
// bundle. Secrets are already redacted in Manifest, so the bytes are publish-safe.
func (a ReplayArtifact) JSON() []byte {
	b, _ := json.MarshalIndent(a, "", "  ")
	return b
}

// ReplayFrom reconstructs an artifact from JSON emitted by JSON(). It is the
// "independently replayed environment" entry point: a fresh process holding only
// the bundle can rebuild the manifest and re-derive its Fingerprint.
func ReplayFrom(data []byte) (ReplayArtifact, error) {
	var a ReplayArtifact
	if err := json.Unmarshal(data, &a); err != nil {
		return ReplayArtifact{}, err
	}
	return a, nil
}
