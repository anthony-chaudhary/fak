package trajectory

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuditUnknownExemplarsCoalesceLinkAndRedact(t *testing.T) {
	const secret = "sk-ant-api03-DO-NOT-PERSIST"
	const sensitiveType = "customer-project-blue"
	d := newAuditDistribution()
	visible := []byte(`{"type":"response_item","payload":{"type":"` + sensitiveType + `","secret":"` + secret + `","items":[{"value":"first"}]}}`)
	visibleRepeat := []byte(`{"type":"response_item","payload":{"type":"` + sensitiveType + `","secret":"different-secret","items":[{"value":"second"}]}}`)
	storage := []byte(`{"type":"` + secret + `","payload":{"password":"` + secret + `","enabled":true}}`)
	d.observe(AuditSourceCodex, visible)
	d.observe(AuditSourceCodex, visibleRepeat)
	d.observe(AuditSourceCodex, storage)

	exemplars := d.exemplars.snapshot()
	if len(exemplars.Exemplars) != 2 {
		t.Fatalf("exemplars = %+v, want one coalesced visible shape and one storage shape", exemplars)
	}
	if exemplars.CardinalityLimit != AuditUnknownExemplarMaxCount || exemplars.ByteLimit != AuditUnknownExemplarMaxBytes {
		t.Fatalf("limits = %d/%d", exemplars.CardinalityLimit, exemplars.ByteLimit)
	}
	if got := exemplarByVisibility(t, exemplars.Exemplars, "visible_unknown"); got.Observations != 2 {
		t.Fatalf("visible observations = %d, want 2", got.Observations)
	} else if !strings.Contains(got.Structure, "secret\":string") {
		t.Fatalf("visible structure does not retain field name/type: %+v", got)
	}
	for _, exemplar := range exemplars.Exemplars {
		if exemplar.ID == "" || exemplar.ShapeHash == "" || !strings.Contains(exemplar.Structure, ":string") {
			t.Fatalf("structural exemplar is not classifiable: %+v", exemplar)
		}
	}

	distribution := d.distributionRows()
	visibleRow := distributionRowByName(t, distribution, "visible_unknown")
	if len(visibleRow.ExemplarIDs) != 1 || visibleRow.ExemplarIDs[0] != exemplarByVisibility(t, exemplars.Exemplars, "visible_unknown").ID {
		t.Fatalf("visible exemplar links = %v", visibleRow.ExemplarIDs)
	}
	storageRows := d.storageRows()
	if len(storageRows) != 1 || len(storageRows[0].ExemplarIDs) != 1 || storageRows[0].ExemplarIDs[0] != exemplarByVisibility(t, exemplars.Exemplars, "storage_unknown").ID {
		t.Fatalf("storage exemplar links = %+v", storageRows)
	}

	result := AuditResult{Summary: AuditSummaryRow{
		Schema: AuditSchema, Kind: "summary", Distribution: distribution,
		StorageDistribution: storageRows, UnknownExemplars: exemplars,
	}}
	var jsonl, markdown bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	for name, persisted := range map[string]string{"jsonl": jsonl.String(), "markdown": markdown.String()} {
		if strings.Contains(persisted, secret) || strings.Contains(persisted, sensitiveType) || strings.Contains(persisted, "different-secret") || strings.Contains(persisted, "first") || strings.Contains(persisted, "second") {
			t.Fatalf("%s leaked payload content: %s", name, persisted)
		}
		for _, exemplar := range exemplars.Exemplars {
			if !strings.Contains(persisted, exemplar.ID) {
				t.Fatalf("%s does not link exemplar %s: %s", name, exemplar.ID, persisted)
			}
		}
	}
}

func TestAuditUnknownExemplarReservoirHasIndependentCardinalityAndByteBounds(t *testing.T) {
	t.Run("cardinality", func(t *testing.T) {
		r := newAuditUnknownExemplarReservoir(3, 1<<20)
		for i := 0; i < 50; i++ {
			line := []byte(fmt.Sprintf(`{"type":"future","payload":{"field_%02d":true}}`, i))
			r.observe(AuditSourceCodex, "future", "storage_unknown", "future", line, int64(len(line)))
		}
		got := r.snapshot()
		if len(got.Exemplars) != 3 || got.Retained != 3 || got.DroppedObservations == 0 {
			t.Fatalf("cardinality reservoir = %+v", got)
		}
		assertSerializedExemplarBound(t, got)
	})

	t.Run("serialized bytes", func(t *testing.T) {
		const byteLimit = 900
		r := newAuditUnknownExemplarReservoir(100, byteLimit)
		for i := 0; i < 50; i++ {
			line := []byte(fmt.Sprintf(`{"type":"future","payload":{"field_%02d_%s":true}}`, i, strings.Repeat("x", 100)))
			r.observe(AuditSourceCodex, "future", "storage_unknown", "future", line, int64(len(line)))
		}
		got := r.snapshot()
		if len(got.Exemplars) >= 50 || got.StoredBytes > byteLimit || got.DroppedObservations == 0 {
			t.Fatalf("byte reservoir = %+v", got)
		}
		assertSerializedExemplarBound(t, got)
	})
}

func TestAuditUnknownExemplarIdentityIncludesClassificationCase(t *testing.T) {
	r := newAuditUnknownExemplarReservoir(10, 1<<20)
	line := []byte(`{"type":"future","payload":{"value":"discarded"}}`)
	r.observe(AuditSourceCodex, "future/a", "visible_unknown", "visible_unknown", line, 10)
	r.observe(AuditSourceCodex, "future/b", "visible_unknown", "visible_unknown", line, 10)
	r.observe(AuditSourceCodex, "future/a", "storage_unknown", "future/a", line, 10)
	got := r.snapshot()
	if len(got.Exemplars) != 3 {
		t.Fatalf("classification cases coalesced: %+v", got.Exemplars)
	}
	wantShape := got.Exemplars[0].ShapeHash
	ids := map[string]bool{}
	for _, exemplar := range got.Exemplars {
		if exemplar.ShapeHash != wantShape {
			t.Fatalf("identical JSON shape has different shape hashes: %+v", got.Exemplars)
		}
		if ids[exemplar.ID] {
			t.Fatalf("classification cases share ID %q: %+v", exemplar.ID, got.Exemplars)
		}
		ids[exemplar.ID] = true
	}
}

func TestAuditUnknownExemplarsCoverClaudeSubtypeAndVisibilityCases(t *testing.T) {
	d := newAuditDistribution()
	d.observe(AuditSourceClaude, []byte(`{"type":"assistant","message":{"content":[{"type":"customer_alpha","private":"discard-me"}]}}`))
	d.observe(AuditSourceClaude, []byte(`{"type":"assistant","message":{"content":[{"type":"customer_beta","private":"discard-me"}]}}`))
	d.observe(AuditSourceClaude, []byte(`{"type":"attachment","attachment":{"type":"future_attachment","private":"discard-me"}}`))

	got := d.exemplars.snapshot()
	if len(got.Exemplars) != 3 {
		t.Fatalf("Claude unknown cases = %+v", got.Exemplars)
	}
	visible := distributionRowByName(t, d.distributionRows(), "visible_unknown")
	if len(visible.ExemplarIDs) != 2 {
		t.Fatalf("Claude unknown block links = %v", visible.ExemplarIDs)
	}
	storage := d.storageRows()
	if len(storage) != 1 || !strings.HasPrefix(storage[0].Subtype, "attachment/discriminator#") || len(storage[0].ExemplarIDs) != 1 {
		t.Fatalf("Claude unknown attachment links = %+v", storage)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("discard-me")) || bytes.Contains(encoded, []byte("customer_alpha")) || bytes.Contains(encoded, []byte("customer_beta")) || bytes.Contains(encoded, []byte("future_attachment")) {
		t.Fatalf("Claude exemplar retained payload: %s", encoded)
	}
	visibleExemplars := make([]AuditUnknownExemplar, 0, 2)
	for _, exemplar := range got.Exemplars {
		if exemplar.Visibility == "visible_unknown" {
			visibleExemplars = append(visibleExemplars, exemplar)
		}
	}
	if len(visibleExemplars) != 2 || visibleExemplars[0].ID == visibleExemplars[1].ID || visibleExemplars[0].ShapeHash != visibleExemplars[1].ShapeHash {
		t.Fatalf("Claude discriminator hashes did not distinguish identical structures: %+v", visibleExemplars)
	}
}

func TestAuditUnknownExemplarsCoverNestedCodexSubtypes(t *testing.T) {
	d := newAuditDistribution()
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"customer_alpha","text":"discard-one"}]}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"customer_beta","text":"discard-two"}]}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"event_msg","payload":{"type":"item_started","item":{"type":"project_alpha","state":"private"}}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"event_msg","payload":{"type":"item_started","item":{"type":"project_beta","state":"private"}}}`))

	got := d.exemplars.snapshot()
	if len(got.Exemplars) != 4 {
		t.Fatalf("nested Codex unknown cases = %+v", got.Exemplars)
	}
	visible := distributionRowByName(t, d.distributionRows(), "visible_unknown")
	if len(visible.ExemplarIDs) != 2 {
		t.Fatalf("nested Codex content links = %v", visible.ExemplarIDs)
	}
	storage := d.storageRows()
	if len(storage) != 2 {
		t.Fatalf("nested Codex item rows = %+v", storage)
	}
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"customer_alpha", "customer_beta", "project_alpha", "project_beta", "discard-one", "discard-two", "private"} {
		if bytes.Contains(encoded, []byte(forbidden)) {
			t.Fatalf("nested discriminator/payload %q leaked: %s", forbidden, encoded)
		}
	}
}

func TestAuditUnknownExemplarCanonicalDuplicateFragmentDoesNotInflate(t *testing.T) {
	d := newAuditDistribution()
	d.observe(AuditSourceCodex, []byte(`{"type":"future","payload":{"value":"discarded"}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"future_visible","value":"discarded"}}`))
	row := AuditTranscriptRow{
		Schema: AuditSchema, Kind: "session", Source: AuditSourceCodex, TranscriptID: "duplicate-session",
		SourcePath: "one.jsonl", Distribution: d.distributionRows(),
		ToolDistribution:    []AuditDistributionRow{{Name: "exec_command", Bytes: 17, Share: 1, Calls: 1}},
		ToolResults:         []AuditToolResultRow{{Name: "exec_command", Subtype: "command", Bytes: 23, Results: 1, Success: 1}},
		StorageDistribution: d.storageRows(),
		UnknownExemplars:    d.exemplars.snapshot(), fragmentDigest: "same-content-digest",
		ToolCalls: 3, ToolErrors: 2, RepeatedFailures: 4, ExpectedWaitTimeouts: 5, MutationChurn: 6,
	}
	duplicate := row
	duplicate.SourcePath = "two.jsonl"
	canonical, refusals := canonicalAuditTranscripts([]AuditTranscriptRow{row, duplicate})
	if len(refusals) != 0 || len(canonical) != 1 {
		t.Fatalf("canonical = %+v, refusals = %+v", canonical, refusals)
	}
	got := canonical[0]
	if len(got.StorageDistribution) != 1 || got.StorageDistribution[0].Records != 1 || got.StorageDistribution[0].Bytes != row.StorageDistribution[0].Bytes {
		t.Fatalf("duplicate fragment inflated storage: %+v", got.StorageDistribution)
	}
	if len(got.Distribution) != 1 || got.Distribution[0].Bytes != row.Distribution[0].Bytes {
		t.Fatalf("duplicate fragment inflated visible distribution: %+v", got.Distribution)
	}
	if len(got.ToolDistribution) != 1 || got.ToolDistribution[0].Bytes != row.ToolDistribution[0].Bytes || got.ToolDistribution[0].Calls != row.ToolDistribution[0].Calls {
		t.Fatalf("duplicate fragment inflated tool distribution: %+v", got.ToolDistribution)
	}
	if len(got.ToolResults) != 1 || got.ToolResults[0] != row.ToolResults[0] {
		t.Fatalf("duplicate fragment inflated tool-result projection: %+v", got.ToolResults)
	}
	if len(got.UnknownExemplars.Exemplars) != 2 {
		t.Fatalf("duplicate fragment inflated exemplars: %+v", got.UnknownExemplars)
	}
	for i, exemplar := range got.UnknownExemplars.Exemplars {
		if exemplar.Observations != 1 || exemplar.ObservedBytes != row.UnknownExemplars.Exemplars[i].ObservedBytes {
			t.Fatalf("duplicate fragment inflated exemplar %d: %+v", i, got.UnknownExemplars)
		}
	}
	if got.ToolCalls != row.ToolCalls || got.ToolErrors != row.ToolErrors || got.RepeatedFailures != row.RepeatedFailures || got.ExpectedWaitTimeouts != row.ExpectedWaitTimeouts || got.MutationChurn != row.MutationChurn {
		t.Fatalf("duplicate fragment inflated derived metrics: got=%+v want=%+v", got, row)
	}
}

func TestAuditCanonicalDistinctFragmentsMergeToolDistributionCallsAndBytes(t *testing.T) {
	first := AuditTranscriptRow{
		Schema: AuditSchema, Kind: "session", Source: AuditSourceCodex, TranscriptID: "split-session",
		SourcePath: "one.jsonl", fragmentDigest: "first-fragment",
		ToolDistribution: []AuditDistributionRow{
			{Name: "exec_command", Bytes: 10, Calls: 1},
			{Name: "read_file", Bytes: 5, Calls: 2},
		},
	}
	second := AuditTranscriptRow{
		Schema: AuditSchema, Kind: "session", Source: AuditSourceCodex, TranscriptID: "split-session",
		SourcePath: "two.jsonl", fragmentDigest: "second-fragment",
		ToolDistribution: []AuditDistributionRow{
			{Name: "exec_command", Bytes: 20, Calls: 3},
			{Name: "write_file", Bytes: 7, Calls: 1},
		},
	}

	canonical, refusals := canonicalAuditTranscripts([]AuditTranscriptRow{first, second})
	if len(refusals) != 0 || len(canonical) != 1 {
		t.Fatalf("canonical = %+v, refusals = %+v", canonical, refusals)
	}
	rows := canonical[0].ToolDistribution
	for name, want := range map[string]struct {
		bytes int64
		calls int
	}{
		"exec_command": {bytes: 30, calls: 4},
		"read_file":    {bytes: 5, calls: 2},
		"write_file":   {bytes: 7, calls: 1},
	} {
		got := distributionRowByName(t, rows, name)
		if got.Bytes != want.bytes || got.Calls != want.calls {
			t.Fatalf("%s merged row = %+v, want bytes=%d calls=%d", name, got, want.bytes, want.calls)
		}
	}
}

func TestRunAuditDuplicateFilesDoNotWeightHookOrToolErrorSideChannels(t *testing.T) {
	root := t.TempDir()
	lowHooksAndErrors := make([]string, 0, 14)
	for i := 0; i < 10; i++ {
		lowHooksAndErrors = append(lowHooksAndErrors, `{"sessionId":"duplicate-session","type":"attachment","attachment":{"type":"hook_success","durationMs":10}}`)
	}
	lowHooksAndErrors = append(lowHooksAndErrors,
		`{"sessionId":"duplicate-session","type":"assistant","message":{"id":"usage-one","usage":{"input_tokens":10,"output_tokens":0},"content":[{"type":"tool_use","id":"edit-one","name":"edit_file","input":{"path":"fixture.go"}}]}}`,
		`{"sessionId":"duplicate-session","type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"edit-one","is_error":true,"content":"permission denied"}]}}`,
		`{"sessionId":"duplicate-session","type":"assistant","message":{"id":"usage-two","usage":{"input_tokens":10,"output_tokens":0},"content":[{"type":"tool_use","id":"edit-two","name":"edit_file","input":{"path":"fixture.go"}}]}}`,
		`{"sessionId":"duplicate-session","type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"edit-two","is_error":true,"content":"permission denied"}]}}`,
	)
	lowFragment := []byte(strings.Join(lowHooksAndErrors, "\n") + "\n")
	highFragment := []byte(`{"sessionId":"duplicate-session","type":"attachment","attachment":{"type":"hook_success","durationMs":100}}` + "\n")
	for name, content := range map[string][]byte{
		"a-low.jsonl":      lowFragment,
		"b-low-copy.jsonl": lowFragment,
		"c-high.jsonl":     highFragment,
	} {
		if err := os.WriteFile(filepath.Join(root, name), content, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	result, err := RunAudit(AuditOptions{Sources: []AuditSource{{Name: AuditSourceClaude, Root: root, RootLabel: "duplicate-fixture"}}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Summary.HookP95MS == nil || *result.Summary.HookP95MS != 100 {
		t.Fatalf("duplicate low-hook fragment weighted p95: %v, want 100", result.Summary.HookP95MS)
	}
	if len(result.ToolErrorFamilies) != 1 {
		t.Fatalf("tool-error families = %+v, want one", result.ToolErrorFamilies)
	}
	family := result.ToolErrorFamilies[0]
	if family.Family != "permission" || family.Count != 2 || family.Tokens != 0 || family.RepeatedFailures != 1 || family.MutationChurn != 1 {
		t.Fatalf("duplicate fragment inflated tool-error family totals: %+v", family)
	}
	if result.Summary.ToolErrors != 2 || result.Summary.MutationChurn != 1 || result.Summary.RepeatedFailures != 1 {
		t.Fatalf("canonical summary was inflated: errors=%d churn=%d repeats=%d", result.Summary.ToolErrors, result.Summary.MutationChurn, result.Summary.RepeatedFailures)
	}
}

func TestAuditUnknownCodexMessageRoleIsOpaqueAndLinked(t *testing.T) {
	const privateRole = "customer-project-blue"
	d := newAuditDistribution()
	d.observe(AuditSourceCodex, []byte(`{"type":"response_item","payload":{"type":"message","role":"`+privateRole+`","content":[{"type":"input_text","text":"discard-me"}]}}`))
	rows := d.distributionRows()
	unknown := distributionRowByName(t, rows, "visible_unknown")
	if len(unknown.ExemplarIDs) != 1 {
		t.Fatalf("unknown role exemplar links = %v", unknown.ExemplarIDs)
	}
	got := d.exemplars.snapshot()
	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte(privateRole)) || bytes.Contains(encoded, []byte("discard-me")) {
		t.Fatalf("unknown role leaked into exemplar: %s", encoded)
	}
	if len(got.Exemplars) != 1 || !strings.HasPrefix(got.Exemplars[0].Subtype, "response_item/discriminator#") {
		t.Fatalf("unknown role subtype = %+v", got.Exemplars)
	}
}

func TestAuditUnknownSourceIsOpaqueAndLinked(t *testing.T) {
	const privateSource = "customer-project-source"
	d := newAuditDistribution()
	d.observe(privateSource, []byte(`{"type":"future","payload":{"value":"discard-me"}}`))
	storage := d.storageRows()
	if len(storage) != 1 || strings.Contains(storage[0].Source, privateSource) || !strings.HasPrefix(storage[0].Source, "source/discriminator#") || len(storage[0].ExemplarIDs) != 1 {
		t.Fatalf("unknown source row = %+v", storage)
	}
	encoded, err := json.Marshal(struct {
		Storage   []AuditStorageRow             `json:"storage"`
		Exemplars AuditUnknownExemplarReservoir `json:"exemplars"`
	}{storage, d.exemplars.snapshot()})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte(privateSource)) || bytes.Contains(encoded, []byte("discard-me")) {
		t.Fatalf("unknown source leaked into persisted projection: %s", encoded)
	}
}

func TestAuditConformanceFutureDiscriminatorsRemainEphemeral(t *testing.T) {
	d := newAuditDistribution()
	d.observe(AuditSourceCodex, []byte(`{"type":"event_msg","payload":{"type":"future_event_v99","private":"discard-event"}}`))
	d.observe(AuditSourceCodex, []byte(`{"type":"future_row_v99","payload":{"private":"discard-row"}}`))

	storage := d.storageRows()
	wantSubtypes := map[string]bool{
		"event_msg/discriminator#8dde2a3a3ed8": false,
		"record/discriminator#9e88b29883a2":    false,
	}
	for _, row := range storage {
		if _, ok := wantSubtypes[row.Subtype]; !ok {
			t.Fatalf("unexpected persisted subtype: %+v", row)
		}
		wantSubtypes[row.Subtype] = true
	}
	for subtype, seen := range wantSubtypes {
		if !seen {
			t.Fatalf("missing persisted subtype %q in %+v", subtype, storage)
		}
	}

	result := AuditResult{Summary: AuditSummaryRow{
		Schema: AuditSchema, Kind: "summary", StorageDistribution: storage,
		UnknownExemplars: d.exemplars.snapshot(),
	}}
	var jsonl, markdown bytes.Buffer
	if err := WriteAuditJSONL(&jsonl, result); err != nil {
		t.Fatal(err)
	}
	if err := WriteAuditMarkdown(&markdown, result); err != nil {
		t.Fatal(err)
	}
	for name, persisted := range map[string]string{"storage JSON": string(mustJSON(storage)), "JSONL": jsonl.String(), "Markdown": markdown.String()} {
		for _, forbidden := range []string{"future_event_v99", "future_row_v99", "discard-event", "discard-row"} {
			if strings.Contains(persisted, forbidden) {
				t.Fatalf("%s persisted raw future discriminator/payload %q: %s", name, forbidden, persisted)
			}
		}
	}
}

func TestAuditUnknownExemplarSummaryAndCanonicalMergesStayBounded(t *testing.T) {
	rows := make([]AuditTranscriptRow, 0, AuditUnknownExemplarMaxCount*3)
	for i := 0; i < AuditUnknownExemplarMaxCount*3; i++ {
		d := newAuditDistribution()
		line := []byte(fmt.Sprintf(`{"type":"future","payload":{"shape_%03d":"secret-%03d"}}`, i, i))
		d.observe(AuditSourceCodex, line)
		rows = append(rows, AuditTranscriptRow{
			Schema: AuditSchema, Kind: "session", Source: AuditSourceCodex,
			TranscriptID: "same-session", SourcePath: fmt.Sprintf("fragment-%03d.jsonl", i),
			Distribution: d.distributionRows(), StorageDistribution: d.storageRows(), UnknownExemplars: d.exemplars.snapshot(),
		})
	}

	canonical, refusals := canonicalAuditTranscripts(rows)
	if len(refusals) != 0 || len(canonical) != 1 {
		t.Fatalf("canonical = %d, refusals = %+v", len(canonical), refusals)
	}
	assertSerializedExemplarBound(t, canonical[0].UnknownExemplars)
	if canonical[0].UnknownExemplars.Retained > AuditUnknownExemplarMaxCount {
		t.Fatalf("canonical reservoir re-expanded: %+v", canonical[0].UnknownExemplars)
	}
	summary := summarizeAudit(nil, canonical, nil)
	assertSerializedExemplarBound(t, summary.UnknownExemplars)
	if summary.UnknownExemplars.Retained > AuditUnknownExemplarMaxCount {
		t.Fatalf("summary reservoir re-expanded: %+v", summary.UnknownExemplars)
	}
	for _, row := range summary.StorageDistribution {
		for _, id := range row.ExemplarIDs {
			if !hasExemplarID(summary.UnknownExemplars.Exemplars, id) {
				t.Fatalf("summary row links evicted exemplar %q: %+v", id, row)
			}
		}
	}
}

func assertSerializedExemplarBound(t *testing.T, got AuditUnknownExemplarReservoir) {
	t.Helper()
	encoded, err := json.Marshal(got.Exemplars)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != got.StoredBytes {
		t.Fatalf("stored_bytes = %d, serialized exemplars = %d", got.StoredBytes, len(encoded))
	}
	if len(encoded) > got.ByteLimit {
		t.Fatalf("serialized exemplars = %d, byte limit = %d", len(encoded), got.ByteLimit)
	}
}

func exemplarByVisibility(t *testing.T, rows []AuditUnknownExemplar, visibility string) AuditUnknownExemplar {
	t.Helper()
	for _, row := range rows {
		if row.Visibility == visibility {
			return row
		}
	}
	t.Fatalf("missing %s exemplar in %+v", visibility, rows)
	return AuditUnknownExemplar{}
}

func distributionRowByName(t *testing.T, rows []AuditDistributionRow, name string) AuditDistributionRow {
	t.Helper()
	for _, row := range rows {
		if row.Name == name {
			return row
		}
	}
	t.Fatalf("missing %s distribution row in %+v", name, rows)
	return AuditDistributionRow{}
}
