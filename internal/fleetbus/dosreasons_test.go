package fleetbus

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestReasonsRegisteredInDosToml binds the Go closed vocabulary to the workspace
// one. Reasons() is the producer of record; a token that exists only in Go resolves
// as known=false through dos_check_reason and never appears in dos_refuse_reasons,
// which is exactly the UNCLASSIFIED drift a closed vocabulary exists to prevent —
// an operator reading FLEETBUS_APPLY_REFUSED off an ack gets no fix line, and the
// refusal degrades into free text wearing a token's clothes.
//
// It binds all three halves of "known, refusable, actionable": the table must
// exist, declare refusal = true, and carry a non-empty summary AND fix — a
// registered token with an empty fix tells an operator nothing they did not
// already have from the ack.
func TestReasonsRegisteredInDosToml(t *testing.T) {
	content := readRepoDosToml(t)
	for _, r := range Reasons() {
		header := "[reasons." + string(r) + "]"
		if !strings.Contains(content, header) {
			t.Errorf("refusal reason %q has no %s table in dos.toml — dos_check_reason would return known=false", r, header)
			continue
		}
		block := dosReasonBlock(content, header)
		if !reasonFieldTrue(block, "refusal") {
			t.Errorf("refusal reason %q is registered but not marked refusal = true — dos_check_reason would resolve it as non-refusable", r)
		}
		for _, field := range []string{"summary", "fix", "see_also"} {
			if !reasonFieldSet(block, field) {
				t.Errorf("refusal reason %q registration has no non-empty %s — the token would refuse without telling an operator what to do", r, field)
			}
		}
	}
	// The whole group must stay traceable to the issue that opened it, so a later
	// reader can find why five tokens appeared at once.
	if !strings.Contains(content, "#5600") {
		t.Error("dos.toml does not cite issue #5600 — the fleet-bus reason group's provenance is unbound")
	}
}

// TestFleetBusLaneRegisteredInDosToml checks the other half of the registration: a
// leaf whose tree is not claimed by a lane routes nowhere, so its work is invisible
// to dispatch even though the code is on the trunk.
func TestFleetBusLaneRegisteredInDosToml(t *testing.T) {
	content := readRepoDosToml(t)
	if !strings.Contains(content, `fleetbus = ["internal/fleetbus/**"]`) {
		t.Error("dos.toml [lanes.trees] does not claim internal/fleetbus/** — work in this leaf would route to no lane")
	}
	if !strings.Contains(content, `"fleetbus"`) {
		t.Error("dos.toml does not list fleetbus among the leaf names")
	}
}

// dosReasonBlock returns the text of the [reasons.<TOKEN>] table named by header:
// from the header line up to (but excluding) the next top-level [section] or EOF,
// so a field assertion cannot match a sibling table's line by accident.
func dosReasonBlock(content, header string) string {
	i := strings.Index(content, header)
	if i < 0 {
		return ""
	}
	rest := content[i+len(header):]
	if j := strings.Index(rest, "\n["); j >= 0 {
		return content[i : i+len(header)+j]
	}
	return content[i:]
}

// reasonFieldTrue reports whether block contains a `field = true` line, tolerant of
// the aligned whitespace the dos.toml [reasons] tables use ("refusal  = true").
func reasonFieldTrue(block, field string) bool {
	return strings.TrimSpace(reasonFieldValue(block, field)) == "true"
}

// reasonFieldSet reports whether block declares field with a value that is more
// than empty quotes or an empty list.
func reasonFieldSet(block, field string) bool {
	v := strings.TrimSpace(reasonFieldValue(block, field))
	return v != "" && v != `""` && v != "[]"
}

func reasonFieldValue(block, field string) string {
	for _, line := range strings.Split(block, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, field) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(line, field))
		if strings.HasPrefix(rest, "=") {
			return rest[1:]
		}
	}
	return ""
}

// readRepoDosToml reads the repo-root dos.toml relative to this test's own source
// path, so the lookup does not depend on the working directory the suite runs from.
func readRepoDosToml(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed — cannot locate the test source path")
	}
	root := filepath.Join(filepath.Dir(thisFile), "..", "..")
	b, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Fatalf("read repo dos.toml: %v", err)
	}
	return string(b)
}
