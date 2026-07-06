package modelscore

import _ "embed"

// DefaultRegistryPath is the repo-relative path of the embedded fixture (for
// messages / --registry override discovery).
const DefaultRegistryPath = "internal/modelscore/builtin.json"

//go:embed builtin.json
var builtin []byte

// Builtin returns the embedded fixture registry. It demonstrates the SHAPE —
// Terminal-Bench, SWE-bench, and FrontierSWE rows plus cost and context, each
// provenanced — over a few tier stand-in models. Every score is marked
// Illustrative: the numbers are placeholders to be superseded by ingested
// leaderboard evidence, never a measured claim. The binary is self-contained;
// a real deployment overrides it with an ingested snapshot via Load.
func Builtin() (*Registry, error) { return Load(builtin) }
