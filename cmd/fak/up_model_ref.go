package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// resolveUpKVPrecision resolves the realized KV storage tier for `fak up`. Precedence:
// an explicit flag value wins; otherwise FAK_UP_KV_PRECISION; otherwise f32 (the exact
// default). It pins the resolved value back into FAK_UP_KV_PRECISION so the native
// loader dep path and the per-request planner cannot drift. Unknown tokens refuse
// rather than silently falling back, so a typo never buys a lossy cache unnoticed.
func resolveUpKVPrecision(flagValue string) (fakmodel.KVPrecision, error) {
	raw := strings.TrimSpace(flagValue)
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv("FAK_UP_KV_PRECISION"))
	}
	if raw == "" {
		return fakmodel.KVPrecisionFP32, nil
	}
	prec, err := fakmodel.ParseKVPrecision(raw)
	if err != nil {
		return "", err
	}
	_ = os.Setenv("FAK_UP_KV_PRECISION", string(prec))
	return prec, nil
}

// turnkeyStableFitBudget builds the loader admission budget the turnkey path pins
// for a host whose total unified memory is memoryBytes. It is deliberately STABLE:
// the base is the physical total and the headroom is macfit's documented 20% OS/
// other-app reserve, so the number does not move with the instantaneous free-memory
// reading. That is the fix for a fresh-boot/cold-cache box spuriously refusing a
// context the reserve-based envelope admits. memoryBytes==0 (an unknown host, e.g. a
// unit-test profile) yields nil, preserving the historical live probe.
func turnkeyStableFitBudget(memoryBytes uint64) *serveFitBudget {
	if memoryBytes == 0 {
		return nil
	}
	return &serveFitBudget{Base: int64(memoryBytes), Headroom: macfit.DefaultMinHeadroomRatio}
}

func turnkeyContextTokens(tokens uint64) (int, error) {
	maxInt := uint64(^uint(0) >> 1)
	if tokens > maxInt {
		return 0, fmt.Errorf("turnkey context budget %d exceeds the platform integer limit %d", tokens, maxInt)
	}
	return int(tokens), nil
}

// turnkeyCanonicalQuantAlias maps a macfit tier's declared quant to the embedded
// catalog alias that unambiguously names that exact quant. Turnkey tier selection
// resolves to these aliases, NOT to a bare family alias like "qwen38:27b": the bare
// alias is user-shadowable via registry.json, so a user overlay could silently
// re-bind the 27B tier to a different quant (witnessed: a UD-Q2_K_XL overlay
// shadowed "qwen38:27b" while the tier budget still assumed 16 GiB Q4_K_M). The
// quant-qualified alias is not in the user overlay and always names Q4_K_M.
//
// Each alias MUST exist in modelreg.Catalog (see TestTurnkeyTierAliasesAreUnshadowed).
func turnkeyCanonicalQuantAlias(tierName string) string {
	switch strings.ToUpper(strings.TrimSpace(tierName)) {
	case "70B":
		return "qwen38:70b-q4_k_m"
	case "27B":
		return "qwen38:27b-q4_k_m"
	case "7B":
		return "qwen3.8-7b-q4_k_m"
	case "3B":
		return "qwen3.8-3b-q4_k_m"
	}
	return ""
}

func resolveTurnkeyModelRef(ref string) string {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" {
		return turnkeyCanonicalQuantAlias("27B")
	}
	if hfhub.IsURI(trimmed) {
		return trimmed
	}
	if _, err := os.Stat(trimmed); err == nil {
		return trimmed
	}
	switch strings.ToLower(trimmed) {
	case "70b", "qwen3.8-70b-q4_k_m", "qwen3.8-70b", "qwen38-70b", "qwen38:70b", "qwen38:70b-q4_k_m":
		return turnkeyCanonicalQuantAlias("70B")
	case "27b", "qwen3.8-27b-q4_k_m", "qwen3.8-27b", "qwen38-27b", "qwen38:27b", "qwen38:27b-q4_k_m":
		return turnkeyCanonicalQuantAlias("27B")
	case "7b", "qwen3.8-7b-q4_k_m", "qwen3.8-7b":
		return "qwen2.5:7b"
	case "3b", "qwen3.8-3b-q4_k_m", "qwen3.8-3b":
		return "qwen2.5-coder:3b"
	}
	return trimmed
}

// artifactDescriptor is the concrete model artifact a turnkey model ref resolves
// to, reduced to the two facts a tier-consistency gate needs: the quantization it
// names and its on-disk byte size. It is a plain value so the gate is a pure,
// network-free function unit-testable with a fabricated descriptor.
type artifactDescriptor struct {
	Ref       string // alias, hf:// URI, or local path as resolved
	Filename  string // basename of the artifact when known
	Quant     string // quant inferred from Filename (e.g. "Q4_K_M"); "" if unknown
	SizeBytes uint64 // on-disk size when known; 0 when unknown
	Explicit  bool   // operator typed a concrete path/hf:// URI (opinionated input)
}

// turnkeyTierSizeTolerance is how far a resolved artifact's byte size may sit from
// the tier's declared WeightBytes before it is treated as a different artifact. A
// lower bound catches a smaller quant masquerading as the tier default (e.g. a
// 9.3 GiB 2-bit file under a 16 GiB Q4_K_M tier); the upper bound catches a larger
// quant. Both are generous because file-size accounting varies slightly, but a
// full quant rung apart (Q2 vs Q4 ~= 0.58x) is far outside the band.
const turnkeyTierSizeTolerance = 0.25

// inferQuantFromFilename extracts a quantization token from a GGUF artifact name
// (e.g. "Qwen3.8-27B-Q4_K_M.gguf" -> "Q4_K_M", "...UD-Q2_K_XL.gguf" -> "Q2_K"). It
// normalizes the underscore/dash spelling to the K-quant form and returns "" when
// no recognizable quant token is present, so callers only fail on a real signal.
func inferQuantFromFilename(name string) string {
	base := strings.ToUpper(strings.TrimSuffix(filepath.Base(name), ".gguf"))
	base = strings.ReplaceAll(base, "_", "-")
	for _, q := range []string{"IQ1-S", "IQ1-M", "IQ2-XXS", "IQ2-XS", "IQ2-S", "IQ3-XXS", "IQ3-XS", "IQ3-S", "IQ4-XS", "IQ4-NL"} {
		if strings.Contains(base, q) {
			return strings.ReplaceAll(q, "-", "_")
		}
	}
	for _, q := range []string{"BF16", "F16", "F32", "FP8", "FP16"} {
		if strings.Contains(base, q) {
			return q
		}
	}
	for _, digits := range []string{"2", "3", "4", "5", "6", "8"} {
		stem := "Q" + digits
		idx := strings.Index(base, stem)
		if idx < 0 {
			continue
		}
		tail := base[idx:]
		var b strings.Builder
		b.WriteString(stem)
		for _, part := range strings.Split(tail, "-")[1:] {
			switch part {
			case "K", "M", "S", "L", "XL", "XS", "XXS", "NL":
				b.WriteString("_" + part)
			default:
				goto done
			}
		}
	done:
		return b.String()
	}
	return ""
}

// validateTurnkeyArtifact fails loud when the artifact a turnkey ref resolves to
// clearly contradicts the selected macfit tier's declared quant/size. An explicit
// operator-supplied full path or hf:// URI is an opinionated override and always
// passes through. An alias, by contrast, must be tier-consistent - it is the
// turnkey surface, so a mismatch means the alias was mis-bound (e.g. a user
// registry.json overlay shadowing the tier default with a 2-bit artifact).
func validateTurnkeyArtifact(tier macfit.ModelTier, art artifactDescriptor) error {
	if art.Explicit {
		return nil
	}
	if art.Quant != "" && tier.QuantTier != "" && !strings.EqualFold(art.Quant, tier.QuantTier) {
		return fmt.Errorf("turnkey model %q (alias %q) declares quant %s but tier %s requires %s; refusing to load a mismatched artifact (use an explicit .gguf path or hf:// URI to override)",
			art.Filename, art.Ref, art.Quant, tier.Name, tier.QuantTier)
	}
	if art.SizeBytes > 0 && tier.WeightBytes > 0 {
		expected := float64(tier.WeightBytes)
		actual := float64(art.SizeBytes)
		if actual < expected*(1-turnkeyTierSizeTolerance) || actual > expected*(1+turnkeyTierSizeTolerance) {
			return fmt.Errorf("turnkey model %q (alias %q) is %.2f GiB but tier %s budgets %.2f GiB for %s; refusing to load a mismatched artifact (use an explicit .gguf path or hf:// URI to override)",
				art.Filename, art.Ref, actual/float64(macfit.GiB), tier.Name, expected/float64(macfit.GiB), tier.QuantTier)
		}
	}
	return nil
}

// turnkeyRefIsExplicit reports whether the operator supplied a concrete model
// reference (an hf:// URI or an existing file) rather than a friendly tier alias.
// Explicit input is an opinionated override, so the tier-consistency gate passes it
// through; only alias-resolved turnkey refs are held to the tier's declared quant.
func turnkeyRefIsExplicit(ref string) bool {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" || strings.EqualFold(trimmed, "default") {
		return false
	}
	if hfhub.IsURI(trimmed) {
		return true
	}
	if _, err := os.Stat(pathutil.ExpandTilde(trimmed)); err == nil {
		return true
	}
	return false
}

// describeTurnkeyArtifact derives an artifact descriptor from a resolved model ref.
// explicit marks a concrete path/URI the operator typed directly (vs a tier alias),
// which the consistency gate treats as an opinionated override.
func describeTurnkeyArtifact(ref string, explicit bool) artifactDescriptor {
	art := artifactDescriptor{Ref: ref, Explicit: explicit}
	if hfhub.IsURI(ref) {
		if parsed, err := hfhub.ParseURI(ref); err == nil && parsed.File != "" {
			art.Filename = filepath.Base(parsed.File)
		}
	} else {
		art.Filename = filepath.Base(ref)
	}
	art.Quant = inferQuantFromFilename(art.Filename)
	if fi, err := os.Stat(pathutil.ExpandTilde(ref)); err == nil && !fi.IsDir() {
		art.SizeBytes = uint64(fi.Size())
	}
	return art
}
