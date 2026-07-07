package gateway

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/engine"
)

// fakeCapProducer is a wire-neutral stand-in for an engine-specific adapter: it
// satisfies engine.CacheCapabilityProducer without being one of the real
// vLLM/SGLang/llama adapters (#1551–#1553). The gateway read path must work against
// any producer, so the test uses a fake — item 32 defines the shape, not the data.
type fakeCapProducer struct{ cc engine.CacheCapability }

func (f fakeCapProducer) CacheCapability() engine.CacheCapability { return f.cc }

// TestReportEngineCacheCapabilityClosedVocabulary drives the gateway read path with a
// fake producer for every closed-vocabulary verdict and asserts the reported value
// stays in the closed vocabulary — the issue's default/evidence target ("gateway
// reports what the upstream engine can expose") proven without importing any
// engine-specific package.
func TestReportEngineCacheCapabilityClosedVocabulary(t *testing.T) {
	for _, v := range engine.CacheVerdicts() {
		cc := engine.CacheCapability{
			Engine:          "fake-engine",
			Verdict:         v,
			Provenance:      engine.ProvenanceProvider,
			Evidence:        "unit-test",
			ColdPathCorrect: true, // so active verdicts are not demoted
		}
		got := ReportEngineCacheCapability(fakeCapProducer{cc})
		if !got.Verdict.Valid() {
			t.Errorf("reported verdict %q is not in the closed vocabulary", got.Verdict)
		}
		if got.Verdict != v {
			t.Errorf("reported verdict = %q, want %q", got.Verdict, v)
		}
		if !got.Valid() {
			t.Errorf("reported capability is not well-formed: %+v", got)
		}
	}
}

// TestReportEngineCacheCapabilityNilProducer: a nil producer reports the safe default,
// never a phantom capability.
func TestReportEngineCacheCapabilityNilProducer(t *testing.T) {
	got := ReportEngineCacheCapability(nil)
	if got.Verdict != engine.CacheUnknown {
		t.Errorf("nil producer verdict = %q, want %q", got.Verdict, engine.CacheUnknown)
	}
	if got.Provenance != engine.ProvenanceKernel {
		t.Errorf("nil producer provenance = %q, want %q", got.Provenance, engine.ProvenanceKernel)
	}
	if !got.Valid() {
		t.Errorf("nil producer report is not well-formed: %+v", got)
	}
}

// TestReportEngineCacheCapabilityFailsClosed: an adapter returning an out-of-vocabulary
// verdict or provenance is reported as the safe default, not trusted verbatim.
func TestReportEngineCacheCapabilityFailsClosed(t *testing.T) {
	got := ReportEngineCacheCapability(fakeCapProducer{engine.CacheCapability{
		Verdict:    "totally-bogus",
		Provenance: "totally-bogus",
	}})
	if got.Verdict != engine.CacheUnknown {
		t.Errorf("bogus verdict reported as %q, want %q", got.Verdict, engine.CacheUnknown)
	}
	if got.Provenance != engine.ProvenanceKernel {
		t.Errorf("bogus provenance reported as %q, want %q", got.Provenance, engine.ProvenanceKernel)
	}
}

// TestReportEngineCacheCapabilityColdPathGate: an ACTIVE verdict whose cold path is
// not witnessed correct is demoted to unknown; the same verdict with a witnessed
// cold path is reported as-is. Cold-path correctness stays explicit for active
// behavior.
func TestReportEngineCacheCapabilityColdPathGate(t *testing.T) {
	for _, v := range engine.CacheVerdicts() {
		if !v.Active() {
			continue
		}
		unproven := ReportEngineCacheCapability(fakeCapProducer{engine.CacheCapability{
			Verdict: v, Provenance: engine.ProvenanceKernel, ColdPathCorrect: false,
		}})
		if unproven.Verdict != engine.CacheUnknown {
			t.Errorf("active %q without cold-path witness reported as %q, want %q",
				v, unproven.Verdict, engine.CacheUnknown)
		}
		proven := ReportEngineCacheCapability(fakeCapProducer{engine.CacheCapability{
			Verdict: v, Provenance: engine.ProvenanceKernel, ColdPathCorrect: true,
		}})
		if proven.Verdict != v {
			t.Errorf("active %q with cold-path witness reported as %q, want %q",
				v, proven.Verdict, v)
		}
	}
}

// TestReportEngineCacheCapabilityIsWireNeutral is the import-hygiene witness for item
// 32: the gateway's cache-capability read path reports what an engine can expose
// WITHOUT importing an engine-specific package into core. It parses the read-path
// source and asserts its imports include the wire-neutral engine contract but none of
// the engine-specific adapter packages (internal/enginecache's SGLang/vLLM client, or
// the #1551–#1553 adapters). Fails before enginecapability.go exists; passes after.
func TestReportEngineCacheCapabilityIsWireNeutral(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed; cannot locate the read-path source")
	}
	readPath := filepath.Join(filepath.Dir(thisFile), "enginecapability.go")

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, readPath, nil, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", readPath, err)
	}

	imports := make([]string, 0, len(f.Imports))
	for _, spec := range f.Imports {
		p, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			t.Fatalf("bad import literal %q: %v", spec.Path.Value, err)
		}
		imports = append(imports, p)
	}

	// The read path MUST consume the wire-neutral contract...
	const contract = "github.com/anthony-chaudhary/fak/internal/engine"
	sawContract := false
	for _, p := range imports {
		if p == contract {
			sawContract = true
		}
	}
	if !sawContract {
		t.Errorf("read path does not import the wire-neutral contract %q; imports=%v", contract, imports)
	}

	// ...and MUST NOT import any engine-specific adapter package. The forbidden set is
	// the current SGLang/vLLM control client plus the per-engine adapters that will
	// land as #1551–#1553; a substring match catches them whatever they are named.
	forbidden := []string{
		"internal/enginecache",
		"internal/vllmcache",
		"internal/sglangcache",
		"internal/llamacache",
	}
	for _, p := range imports {
		for _, bad := range forbidden {
			if strings.Contains(p, bad) {
				t.Errorf("read path imports engine-specific adapter package %q (matched %q) — not wire-neutral", p, bad)
			}
		}
	}
}
