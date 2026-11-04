package reachdelta

import (
	"fmt"
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

var (
	benchDeltaLenSink  int
	benchVerdictSink   RatifyVerdict
	benchPromotedSink  adjudicator.Policy
	benchLedgerLenSink int
)

// benchNowUnix is the fixed reference time for every ledger fixture in this
// file; determinism means no clock reads anywhere in the suite.
const benchNowUnix = int64(1762000000)

// benchBasePolicy builds one deterministic, realistically sized base policy
// shared by all benchmarks here. Every name is generated with Sprintf so the
// fixtures are stable across runs.
func benchBasePolicy() adjudicator.Policy {
	const (
		nAllow    = 512
		nDeny     = 256
		nPrefix   = 32
		nHosts    = 128
		nGlobs    = 64
		nPreds    = 256
		nRedact   = 128
		nSecrets  = 256
		nEgressDn = 32
		nComplain = 16
	)
	allow := make(map[string]bool, nAllow)
	for i := 0; i < nAllow; i++ {
		allow[fmt.Sprintf("tool_%03d", i)] = true
	}
	deny := make(map[string]abi.ReasonCode, nDeny)
	for i := 0; i < nDeny; i++ {
		reason := abi.ReasonPolicyBlock
		if i%4 == 0 {
			reason = abi.ReasonDefaultDeny
		}
		deny[fmt.Sprintf("denied_%03d", i)] = reason
	}
	prefixes := make([]string, 0, nPrefix)
	for i := 0; i < nPrefix; i++ {
		prefixes = append(prefixes, fmt.Sprintf("prefix_%02d_", i))
	}
	hosts := make([]string, 0, nHosts)
	for i := 0; i < nHosts; i++ {
		hosts = append(hosts, fmt.Sprintf("host%03d.example.com", i))
	}
	globs := make([]string, 0, nGlobs)
	for i := 0; i < nGlobs; i++ {
		globs = append(globs, fmt.Sprintf("glob_%03d/", i))
	}
	// ONE precompiled regex shared by every predicate; argRuleIDs only ever
	// calls Re.String(), so sharing keeps setup cheap and deterministic.
	sharedRe := regexp.MustCompile(`bench-deny-[0-9]{4}`)
	preds := make([]adjudicator.ArgPredicate, 0, nPreds)
	for i := 0; i < nPreds; i++ {
		preds = append(preds, adjudicator.ArgPredicate{
			Tool:   fmt.Sprintf("pred_tool_%03d", i),
			Arg:    "command",
			Kind:   adjudicator.ArgDenyRegex,
			Re:     sharedRe,
			Reason: abi.ReasonPolicyBlock,
		})
	}
	redact := make([]string, 0, nRedact)
	for i := 0; i < nRedact; i++ {
		redact = append(redact, fmt.Sprintf("field_%03d", i))
	}
	secrets := make([]*regexp.Regexp, 0, nSecrets)
	for i := 0; i < nSecrets; i++ {
		secrets = append(secrets, regexp.MustCompile(fmt.Sprintf(`sk_bench%d_[A-Za-z0-9]{8}`, i)))
	}
	egressDeny := make([]string, 0, nEgressDn)
	for i := 0; i < nEgressDn; i++ {
		egressDeny = append(egressDeny, fmt.Sprintf("denyhost%02d.internal", i))
	}
	complain := make(map[string]bool, nComplain)
	for i := 0; i < nComplain; i++ {
		complain[fmt.Sprintf("complain_%02d", i)] = true
	}
	return adjudicator.Policy{
		Posture:                  adjudicator.PostureFailClosed,
		Allow:                    allow,
		AllowPrefix:              prefixes,
		Deny:                     deny,
		SelfModifyGlobs:          globs,
		ArgPredicates:            preds,
		RedactFields:             redact,
		EgressExtraDenyHosts:     egressDeny,
		ResearchEgressAllowHosts: hosts,
		Complain:                 complain,
		AdvisoryReasons: map[abi.ReasonCode]bool{
			abi.ReasonMalformed:   true,
			abi.ReasonDefaultDeny: true,
			abi.ReasonUnknownTool: true,
		},
		SecretPosture:  adjudicator.SecretQuarantine,
		SecretPatterns: secrets,
		LintWrites:     true,
	}
}

// benchWidenedPolicy copies base and widens a subset of every surface so
// Delta exercises every category: new tools/prefixes/hosts, posture widened to
// AdmitAndLog, secret posture widened, some protections removed, and new
// complain/advisory entries.
func benchWidenedPolicy(base adjudicator.Policy) adjudicator.Policy {
	p := base
	p.Allow = make(map[string]bool, len(base.Allow)+64)
	for k, v := range base.Allow {
		p.Allow[k] = v
	}
	for i := 0; i < 64; i++ {
		p.Allow[fmt.Sprintf("newtool_%03d", i)] = true
	}
	p.Deny = make(map[string]abi.ReasonCode, len(base.Deny)-32)
	for k, v := range base.Deny {
		p.Deny[k] = v
	}
	for i := 0; i < 32; i++ {
		delete(p.Deny, fmt.Sprintf("denied_%03d", i))
	}
	p.AllowPrefix = append([]string(nil), base.AllowPrefix...)
	for i := 0; i < 8; i++ {
		p.AllowPrefix = append(p.AllowPrefix, fmt.Sprintf("newprefix_%02d_", i))
	}
	p.ResearchEgressAllowHosts = append([]string(nil), base.ResearchEgressAllowHosts...)
	for i := 0; i < 16; i++ {
		p.ResearchEgressAllowHosts = append(p.ResearchEgressAllowHosts, fmt.Sprintf("newhost%02d.example.org", i))
	}
	// Tail-slice removals share backing arrays but nothing ever mutates them.
	p.SelfModifyGlobs = base.SelfModifyGlobs[:len(base.SelfModifyGlobs)-16]
	p.ArgPredicates = base.ArgPredicates[:len(base.ArgPredicates)-32]
	p.RedactFields = base.RedactFields[:len(base.RedactFields)-16]
	p.EgressExtraDenyHosts = base.EgressExtraDenyHosts[:len(base.EgressExtraDenyHosts)-8]
	p.SecretPatterns = base.SecretPatterns[:len(base.SecretPatterns)-16]
	p.Complain = make(map[string]bool, len(base.Complain)+8)
	for k, v := range base.Complain {
		p.Complain[k] = v
	}
	for i := 0; i < 8; i++ {
		p.Complain[fmt.Sprintf("trial_%02d", i)] = true
	}
	p.AdvisoryReasons = make(map[abi.ReasonCode]bool, len(base.AdvisoryReasons)+2)
	for k, v := range base.AdvisoryReasons {
		p.AdvisoryReasons[k] = v
	}
	p.AdvisoryReasons[abi.ReasonSelfModify] = true
	p.AdvisoryReasons[abi.ReasonPromptInjection] = true
	p.Posture = adjudicator.PostureAdmitAndLog
	p.SecretPosture = adjudicator.SecretAdmitAndLog
	p.LintWrites = false
	return p
}

// benchAcceptedRiskLedger builds ~1024 knownbad records: live accepted-risk
// rows covering exactly half of findings (so SuppressAccepted drops them),
// plus decoy rows with distinct signatures that must NOT suppress anything
// (resolved status, wrong class, or lapsed TTL).
func benchAcceptedRiskLedger(findings []Finding, nowUnix int64) []knownbad.Record {
	const total = 1024
	liveCount := len(findings) / 2
	recs := make([]knownbad.Record, 0, total)
	for i := 0; i < liveCount; i++ {
		recs = append(recs, knownbad.Record{
			Signature:        findings[i].Signature(),
			ReasonClass:      "accepted-risk",
			Status:           knownbad.StatusOpen,
			DiscoveredAtUnix: nowUnix - 3600,
			TTLSeconds:       86400,
		})
	}
	for i := liveCount; i < total; i++ {
		sig := fmt.Sprintf("reachdelta:new_tool_permitted:decoy_%03d", i-liveCount)
		switch i % 3 {
		case 0:
			recs = append(recs, knownbad.Record{
				Signature: sig, ReasonClass: "accepted-risk", Status: knownbad.StatusResolved,
				DiscoveredAtUnix: nowUnix - 7200, TTLSeconds: 86400,
			})
		case 1:
			recs = append(recs, knownbad.Record{
				Signature: sig, ReasonClass: "crash", Status: knownbad.StatusOpen,
				DiscoveredAtUnix: nowUnix - 3600, TTLSeconds: 86400,
			})
		default:
			recs = append(recs, knownbad.Record{
				Signature: sig, ReasonClass: "accepted-risk", Status: knownbad.StatusOpen,
				DiscoveredAtUnix: nowUnix - 100000, TTLSeconds: 60,
			})
		}
	}
	return recs
}

func BenchmarkDeltaLargePolicy(b *testing.B) {
	base := benchBasePolicy()
	proposed := benchWidenedPolicy(base)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDeltaLenSink = len(Delta(base, proposed))
	}
}

func BenchmarkRatifyWithAcceptedRiskLedger(b *testing.B) {
	base := benchBasePolicy()
	proposed := benchWidenedPolicy(base)
	findings := Delta(base, proposed)
	ledger := benchAcceptedRiskLedger(findings, benchNowUnix)
	benchLedgerLenSink = len(ledger)
	if got, want := len(SuppressAccepted(findings, ledger, benchNowUnix)), len(findings)-len(findings)/2; got != want {
		b.Fatalf("setup: suppressed %d findings, want %d", got, want)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = Ratify(true, base, proposed, ledger, benchNowUnix)
	}
	if benchVerdictSink.Approved || benchVerdictSink.Reason != "reach_expansion" {
		b.Fatalf("verdict = %#v, want unapproved reach_expansion", benchVerdictSink)
	}
}

func BenchmarkRatifyPromotionCloneAndCompose(b *testing.B) {
	base := benchBasePolicy()
	offer := adjudicator.PromotionOffer{Tool: "complain_00", CleanEvents: 10}
	ledger := benchAcceptedRiskLedger(Delta(base, benchWidenedPolicy(base)), benchNowUnix)
	before := struct {
		allow, complain, deny, preds int
		posture                      adjudicator.Posture
		lint                         bool
	}{len(base.Allow), len(base.Complain), len(base.Deny), len(base.ArgPredicates), base.Posture, base.LintWrites}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proposed, verdict := RatifyPromotion(true, base, offer, ledger, benchNowUnix)
		benchPromotedSink = proposed
		benchVerdictSink = verdict
	}
	// One-time post-loop check: the clone path must leave base untouched.
	if len(base.Allow) != before.allow || len(base.Complain) != before.complain ||
		len(base.Deny) != before.deny || len(base.ArgPredicates) != before.preds ||
		base.Posture != before.posture || base.LintWrites != before.lint {
		b.Fatal("RatifyPromotion mutated base")
	}
	if base.Allow["complain_00"] || !base.Complain["complain_00"] {
		b.Fatalf("promotion leaked into base: allow=%t complain=%t", base.Allow["complain_00"], base.Complain["complain_00"])
	}
	if !benchVerdictSink.Approved || benchVerdictSink.Reason != "empty_delta" {
		b.Fatalf("verdict = %#v, want approved empty_delta", benchVerdictSink)
	}
}
