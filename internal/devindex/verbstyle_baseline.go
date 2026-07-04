package devindex

// verbStyleBaseline grandfathers the verb-catalog style defects that were present
// when the style gate (#2249) landed, keyed by "<verb>\t<kind>" (see
// VerbStyleViolation.key). It is FROZEN and may only SHRINK: a defect that gets
// fixed must have its entry removed (TestVerbStyleBaselineShrinksOnly refuses a
// stale entry), and a NEW verb or a NEW defect is never added here — it is fixed at
// the source. This is the pythongate-style ratchet the issue asks for: the gate
// reds on anything NOT in this set, so debt can only fall.
//
// Every entry below is an UNCATALOGED verb — a live cmd/fak dispatch case with no
// curated verbManifest entry, so Verbs() hands it the "not yet cataloged" fallback
// synopsis. Retiring one means adding its curated entry to verbManifest (verbs.go)
// and deleting the line here. Curated-synopsis style debt (width/lead/period) is
// NOT grandfathered — those four over-width synopses were fixed in place.
var verbStyleBaseline = map[string]bool{
	"amd-gpu-facts\tuncataloged":              true,
	"backend\tuncataloged":                    true,
	"commit-subject-coverage\tuncataloged":    true,
	"demo\tuncataloged":                       true,
	"dispatch-conservation\tuncataloged":      true,
	"dup\tuncataloged":                        true,
	"fleet-trend\tuncataloged":                true,
	"hooklat\tuncataloged":                    true,
	"intent\tuncataloged":                     true,
	"issue-contract-repair\tuncataloged":      true,
	"llm-d-smoke\tuncataloged":                true,
	"logvault\tuncataloged":                   true,
	"macbench\tuncataloged":                   true,
	"memgate\tuncataloged":                    true,
	"memory-read\tuncataloged":                true,
	"memory-stability-governor\tuncataloged":  true,
	"node-compare\tuncataloged":               true,
	"plan-audit\tuncataloged":                 true,
	"qwen36-node-reports\tuncataloged":        true,
	"qwen36-parity-witness-gate\tuncataloged": true,
	"readme-visual-audit\tuncataloged":        true,
	"score\tuncataloged":                      true,
	"sidecar\tuncataloged":                    true,
	"skill\tuncataloged":                      true,
	"sota-coverage-scorecard\tuncataloged":    true,
	"waiting\tuncataloged":                    true,
	"watchdog\tuncataloged":                   true,
}
