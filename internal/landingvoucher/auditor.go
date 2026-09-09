package landingvoucher

import (
	"context"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// AuditOptions configures the pre-landing invariant audit.
type AuditOptions struct {
	RepoRoot            string
	CandidateSHA        string
	BaseSHA             string
	Lane                string
	LeasedGlobs         []string
	ChangedFiles        []string
	WitnessCommand      string
	WitnessReceipt      string
	HasLockContention   bool
	HasActiveLease      bool
	ABIBreakageDetected bool
	ForbiddenFilesFound []string
}

// AuditLandingInvariants evaluates all 8 pre-landing invariants and, if satisfied,
// mints a signed, tamper-evident LandingVoucher. If any invariant fails, it returns
// an error wrapping ReasonLandingInvariantRefused naming the failing axis and details.
func AuditLandingInvariants(ctx context.Context, opts AuditOptions) (*LandingVoucher, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// 1. SpatialDisjointness: ChangedFiles are all within LeasedGlobs.
	spatialDisjointness := true
	if len(opts.ChangedFiles) > 0 {
		if len(opts.LeasedGlobs) == 0 {
			spatialDisjointness = false
		} else {
			for _, file := range opts.ChangedFiles {
				matched := false
				for _, glob := range opts.LeasedGlobs {
					if matchGlob(glob, file) {
						matched = true
						break
					}
				}
				if !matched {
					spatialDisjointness = false
					break
				}
			}
		}
	}

	// 2. BaseFreshness: BaseSHA is non-empty and CandidateSHA != BaseSHA (forward progression).
	baseFreshness := strings.TrimSpace(opts.BaseSHA) != "" &&
		strings.TrimSpace(opts.CandidateSHA) != "" &&
		opts.CandidateSHA != opts.BaseSHA

	// 3. ABIStability: !opts.ABIBreakageDetected.
	abiStability := !opts.ABIBreakageDetected

	// 4. CleanPathspec: len(ChangedFiles) > 0 and no untracked/unintended files outside declared scope.
	cleanPathspec := len(opts.ChangedFiles) > 0
	if cleanPathspec {
		for _, f := range opts.ChangedFiles {
			trimmed := strings.TrimSpace(f)
			if trimmed == "" || strings.HasPrefix(trimmed, "/") || strings.HasPrefix(trimmed, "\\") || strings.Contains(trimmed, "..") {
				cleanPathspec = false
				break
			}
		}
	}

	// 5. WitnessProof: WitnessReceipt != "" and WitnessCommand != "".
	witnessProof := strings.TrimSpace(opts.WitnessReceipt) != "" &&
		strings.TrimSpace(opts.WitnessCommand) != ""

	// 6. ZeroLockContention: !opts.HasLockContention.
	zeroLockContention := !opts.HasLockContention

	// 7. WriterLeaseValidity: opts.HasActiveLease is true and Lane != "".
	writerLeaseValidity := opts.HasActiveLease && strings.TrimSpace(opts.Lane) != ""

	// 8. BoundaryAdmission: len(opts.ForbiddenFilesFound) == 0 (no forbidden .ps1/.sh scripts or leak tokens).
	boundaryAdmission := len(opts.ForbiddenFilesFound) == 0
	if boundaryAdmission {
		for _, f := range opts.ChangedFiles {
			ext := strings.ToLower(filepath.Ext(f))
			if ext == ".ps1" || ext == ".sh" || ext == ".bat" || ext == ".cmd" {
				boundaryAdmission = false
				break
			}
		}
	}

	checklist := InvariantChecklist{
		SpatialDisjointness: spatialDisjointness,
		BaseFreshness:       baseFreshness,
		ABIStability:        abiStability,
		CleanPathspec:       cleanPathspec,
		WitnessProof:        witnessProof,
		ZeroLockContention:  zeroLockContention,
		WriterLeaseValidity: writerLeaseValidity,
		BoundaryAdmission:   boundaryAdmission,
	}

	if !checklist.AllPassed() {
		var failedAxes []string
		var details []string

		if !checklist.SpatialDisjointness {
			failedAxes = append(failedAxes, "SpatialDisjointness")
			details = append(details, "changed files are not contained within leased globs")
		}
		if !checklist.BaseFreshness {
			failedAxes = append(failedAxes, "BaseFreshness")
			details = append(details, "base SHA empty or candidate SHA equals base SHA (no forward progression)")
		}
		if !checklist.ABIStability {
			failedAxes = append(failedAxes, "ABIStability")
			details = append(details, "ABI breakage detected")
		}
		if !checklist.CleanPathspec {
			failedAxes = append(failedAxes, "CleanPathspec")
			details = append(details, "changed files empty or contains invalid pathspec")
		}
		if !checklist.WitnessProof {
			failedAxes = append(failedAxes, "WitnessProof")
			details = append(details, "witness command or witness receipt is empty")
		}
		if !checklist.ZeroLockContention {
			failedAxes = append(failedAxes, "ZeroLockContention")
			details = append(details, "lock contention detected")
		}
		if !checklist.WriterLeaseValidity {
			failedAxes = append(failedAxes, "WriterLeaseValidity")
			details = append(details, "writer lease is not active or lane is empty")
		}
		if !checklist.BoundaryAdmission {
			failedAxes = append(failedAxes, "BoundaryAdmission")
			details = append(details, "forbidden files found or script boundary violation")
		}

		return nil, fmt.Errorf("%s: invariants failed: [%s] (%s): %w",
			ReasonLandingInvariantRefused,
			strings.Join(failedAxes, ", "),
			strings.Join(details, "; "),
			ErrLandingInvariantRefused,
		)
	}

	now := time.Now().UTC()
	voucher := &LandingVoucher{
		Schema:         VoucherSchemaV1,
		Timestamp:      now,
		CandidateSHA:   opts.CandidateSHA,
		BaseSHA:        opts.BaseSHA,
		Lane:           opts.Lane,
		LeasedGlobs:    opts.LeasedGlobs,
		ChangedFiles:   opts.ChangedFiles,
		WitnessReceipt: opts.WitnessReceipt,
		Invariants:     checklist,
	}
	voucher.Digest = voucher.ComputeDigest()

	return voucher, nil
}

func normalizeSlash(p string) string {
	p = filepath.ToSlash(filepath.Clean(p))
	p = strings.TrimPrefix(p, "./")
	return p
}

func matchGlob(pattern, target string) bool {
	pattern = normalizeSlash(pattern)
	target = normalizeSlash(target)

	if pattern == target {
		return true
	}
	if pattern == "" || target == "" {
		return false
	}
	if pattern == "**" || pattern == "*" {
		return true
	}
	if strings.HasSuffix(pattern, "/**") {
		prefix := strings.TrimSuffix(pattern, "/**")
		if target == prefix || strings.HasPrefix(target, prefix+"/") {
			return true
		}
	}
	if strings.HasSuffix(pattern, "/*") {
		prefix := strings.TrimSuffix(pattern, "/*")
		if targetDir := filepath.ToSlash(filepath.Dir(target)); targetDir == prefix {
			return true
		}
	}
	if strings.HasSuffix(pattern, "/") {
		prefix := strings.TrimSuffix(pattern, "/")
		if target == prefix || strings.HasPrefix(target, prefix+"/") {
			return true
		}
	}
	if ok, _ := filepath.Match(pattern, target); ok {
		return true
	}

	re := compileGlobRegex(pattern)
	return re != nil && re.MatchString(target)
}

var globRegexCache sync.Map

func compileGlobRegex(pattern string) *regexp.Regexp {
	if v, ok := globRegexCache.Load(pattern); ok {
		return v.(*regexp.Regexp)
	}

	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); {
		if pattern[i] == '*' {
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				if i+2 < len(pattern) && pattern[i+2] == '/' {
					b.WriteString("(?:.*/)?")
					i += 3
					continue
				}
				b.WriteString(".*")
				i += 2
				continue
			}
			b.WriteString("[^/]*")
			i++
			continue
		}
		if pattern[i] == '?' {
			b.WriteString("[^/]")
			i++
			continue
		}
		c := pattern[i]
		if strings.ContainsRune(`.+()|[]{}^$\`, rune(c)) {
			b.WriteByte('\\')
		}
		b.WriteByte(c)
		i++
	}
	b.WriteString("$")

	re, err := regexp.Compile(b.String())
	if err != nil {
		return nil
	}
	globRegexCache.Store(pattern, re)
	return re
}
