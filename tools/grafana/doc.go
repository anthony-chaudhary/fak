// Package grafana holds contract tests for the committed Grafana dashboard
// artifacts under tools/grafana/dashboards and the provisioning inputs that
// feed them.
//
// The checks are intentionally test-only: they compare committed JSON against
// the generator (gen_dashboard.py) and against the structure operators depend
// on, so there is no runtime code to export. This file exists because Go
// refuses to build a directory containing only *_test.go files ("no non-test Go
// files"), which made any repo-wide `go build ./...` or the commit validator's
// build phase fail for this package. It carries no behavior.
package grafana
