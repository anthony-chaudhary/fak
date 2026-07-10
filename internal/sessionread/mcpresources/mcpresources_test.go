package mcpresources

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/directory"
	"github.com/anthony-chaudhary/fak/internal/sessionread/query"
	"github.com/anthony-chaudhary/fak/internal/sessionread/screen"
)

const fixtureSecret = "SECRET_MARKER_XYZ"

// fixtureTurns builds one session's turns: a clean user turn, a clean successful
// tool turn, a clean failed tool turn (so KindToolFailures has a real hit), a SEALED
// turn carrying the secret marker in both Text and Bytes (so it would qualify for
// every query kind if the taint screen ever failed to withhold it), and a trailing
// clean assistant turn.
func fixtureTurns() []query.Turn {
	return []query.Turn{
		{Index: 0, Role: "user", Text: "please look at the repo"},
		{Index: 1, Role: "assistant", Tool: "read_file", ToolTerm: true, Files: []string{"a.go"}, Text: "reading a.go"},
		{Index: 2, Role: "assistant", Tool: "run_tests", ToolTerm: true, ToolFailed: true, Text: "tests failed"},
		{
			Index:      3,
			Role:       "assistant",
			Tool:       "secret_tool",
			ToolTerm:   true,
			ToolFailed: true,
			Files:      []string{"secret.txt"},
			Text:       fixtureSecret + " decision about the launch",
			Bytes:      []byte(fixtureSecret + " raw span bytes"),
			Sealed:     true,
		},
		{Index: 4, Role: "assistant", Text: "decided to ship soon"},
	}
}

func fixtureStore(trace string) Store {
	return Store{
		Sessions: map[string]Session{
			trace: {Trace: trace, Turns: fixtureTurns()},
		},
		Directory: directory.Directory(nil, []directory.DriveRow{
			{TraceID: trace, Run: "RUNNING", Priority: 3},
		}, nil),
	}
}

func TestListReadRoundTrip(t *testing.T) {
	const trace = "trace-A"
	store := fixtureStore(trace)

	resources, err := store.ListResources(trace)
	if err != nil {
		t.Fatalf("ListResources: %v", err)
	}
	// directory root + 4 per-session views.
	if len(resources) != 5 {
		t.Fatalf("expected 5 resources, got %d: %+v", len(resources), resources)
	}
	seen := map[string]bool{}
	for _, r := range resources {
		seen[r.URI] = true
		if r.MimeType != "application/json" {
			t.Errorf("resource %s: mimeType = %q, want application/json", r.URI, r.MimeType)
		}
	}
	for _, want := range []string{
		DirectoryURI,
		sessionURI(trace, viewTurns),
		sessionURI(trace, viewContext),
		sessionURI(trace, viewDecisions),
		sessionURI(trace, viewSpans),
	} {
		if !seen[want] {
			t.Errorf("ListResources missing %s", want)
		}
	}

	// Full disclosure lets the owner read every listed URI, including the
	// full-disclosure-only spans view.
	for _, r := range resources {
		res, err := store.ReadResource(r.URI, trace, sessionread.DisclosureFull)
		if err != nil {
			t.Fatalf("ReadResource(%s): %v", r.URI, err)
		}
		if len(res.Contents) != 1 {
			t.Fatalf("ReadResource(%s): got %d contents, want 1", r.URI, len(res.Contents))
		}
		c := res.Contents[0]
		if c.URI != r.URI {
			t.Errorf("ReadResource(%s): content uri = %q", r.URI, c.URI)
		}
		if c.MimeType != "application/json" {
			t.Errorf("ReadResource(%s): mimeType = %q", r.URI, c.MimeType)
		}
		if !json.Valid([]byte(c.Text)) {
			t.Errorf("ReadResource(%s): text is not valid JSON: %s", r.URI, c.Text)
		}
	}
}

func TestScopedToPrincipal(t *testing.T) {
	const owner = "trace-A"
	const other = "trace-B"
	store := fixtureStore(owner)

	// The owner sees its own session's views.
	ownRes, err := store.ListResources(owner)
	if err != nil {
		t.Fatalf("ListResources(owner): %v", err)
	}
	if len(ownRes) != 5 {
		t.Fatalf("owner listing: got %d resources, want 5: %+v", len(ownRes), ownRes)
	}

	// A different principal's listing omits the owner's session entirely (directory
	// root only).
	otherRes, err := store.ListResources(other)
	if err != nil {
		t.Fatalf("ListResources(other): %v", err)
	}
	if len(otherRes) != 1 || otherRes[0].URI != DirectoryURI {
		t.Fatalf("other listing: got %+v, want just the directory root", otherRes)
	}

	// A different principal's read of the owner's turns view is refused
	// READ_SCOPE_DENIED.
	_, err = store.ReadResource(sessionURI(owner, viewTurns), other, sessionread.DisclosureFull)
	if err == nil {
		t.Fatal("cross-principal ReadResource: expected an error, got nil")
	}
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("cross-principal ReadResource: reason = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}

	// The owner itself still reads fine.
	if _, err := store.ReadResource(sessionURI(owner, viewTurns), owner, sessionread.DisclosureFull); err != nil {
		t.Fatalf("owner ReadResource: %v", err)
	}
}

func TestSealedSpanAbsentFromListAndRead(t *testing.T) {
	const trace = "trace-A"
	store := fixtureStore(trace)

	resources, err := store.ListResources(trace)
	if err != nil {
		t.Fatalf("ListResources: %v", err)
	}
	for _, r := range resources {
		if strings.Contains(r.Name, fixtureSecret) || strings.Contains(r.Description, fixtureSecret) || strings.Contains(r.URI, fixtureSecret) {
			t.Fatalf("ListResources leaked the secret marker in %+v", r)
		}
	}

	views := []string{viewTurns, viewContext, viewDecisions, viewSpans}
	levels := []sessionread.Disclosure{
		sessionread.DisclosureMetadata,
		sessionread.DisclosureRedacted,
		sessionread.DisclosureFull,
	}
	for _, view := range views {
		uri := sessionURI(trace, view)
		for _, level := range levels {
			res, err := store.ReadResource(uri, trace, level)
			if err != nil {
				// A disclosure-gated refusal (e.g. spans at metadata/redacted) carries no
				// content at all — nothing to check, and the refusal detail is itself
				// byte-free by screen.Refusal's contract.
				continue
			}
			for _, c := range res.Contents {
				if strings.Contains(c.Text, fixtureSecret) {
					t.Fatalf("ReadResource(%s, disclosure=%s) leaked the secret marker: %s", uri, level, c.Text)
				}
			}
		}
	}

	// The directory read must not leak it either (trivially true — DirectoryRow
	// carries no turn content — but assert it for completeness).
	res, err := store.ReadResource(DirectoryURI, trace, sessionread.DisclosureFull)
	if err != nil {
		t.Fatalf("ReadResource(directory): %v", err)
	}
	for _, c := range res.Contents {
		if strings.Contains(c.Text, fixtureSecret) {
			t.Fatalf("directory read leaked the secret marker: %s", c.Text)
		}
	}
}

func TestFullDisclosureSeparatelyGated(t *testing.T) {
	const trace = "trace-A"
	store := fixtureStore(trace)

	// Owner, but only a metadata grant: the full-disclosure-only spans view refuses
	// READ_SCOPE_DENIED rather than silently downgrading.
	_, err := store.ReadResource(sessionURI(trace, viewSpans), trace, sessionread.DisclosureMetadata)
	if err == nil {
		t.Fatal("spans view at metadata disclosure: expected an error, got nil")
	}
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("spans view at metadata disclosure: reason = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}

	// Redacted still isn't enough.
	_, err = store.ReadResource(sessionURI(trace, viewSpans), trace, sessionread.DisclosureRedacted)
	if err == nil {
		t.Fatal("spans view at redacted disclosure: expected an error, got nil")
	}
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("spans view at redacted disclosure: reason = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}

	// Full disclosure succeeds for the owner.
	if _, err := store.ReadResource(sessionURI(trace, viewSpans), trace, sessionread.DisclosureFull); err != nil {
		t.Fatalf("spans view at full disclosure: %v", err)
	}

	// Meanwhile the turns view (redacted) is satisfied by a redacted grant.
	if _, err := store.ReadResource(sessionURI(trace, viewTurns), trace, sessionread.DisclosureRedacted); err != nil {
		t.Fatalf("turns view at redacted disclosure: %v", err)
	}
}

func TestUnknownURIFaults(t *testing.T) {
	const trace = "trace-A"
	store := fixtureStore(trace)

	cases := []string{
		"session://" + trace + "/bogus", // unrecognized view
		"not-a-session-uri",             // wrong scheme entirely
		"session:///turns",              // empty trace segment
		"session://" + trace,            // missing view segment
		"session://" + trace + "/",      // empty view segment
	}
	for _, uri := range cases {
		_, err := store.ReadResource(uri, trace, sessionread.DisclosureFull)
		if err == nil {
			t.Errorf("ReadResource(%q): expected an error, got nil", uri)
			continue
		}
		if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadUnknownTrace {
			t.Errorf("ReadResource(%q): reason = %q, want %q", uri, reason, sessionread.ReasonReadUnknownTrace)
		}
	}

	// A syntactically valid URI addressing a session the store does not hold.
	_, err := store.ReadResource(sessionURI("no-such-trace", viewTurns), trace, sessionread.DisclosureFull)
	if err == nil {
		t.Fatal("ReadResource(unknown trace): expected an error, got nil")
	}
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadUnknownTrace {
		t.Fatalf("ReadResource(unknown trace): reason = %q, want %q", reason, sessionread.ReasonReadUnknownTrace)
	}
}

func TestDirectoryResource(t *testing.T) {
	const trace = "trace-A"
	dir := directory.Directory(nil, []directory.DriveRow{
		{TraceID: trace, Run: "RUNNING", Priority: 7},
		{TraceID: "trace-other", Run: "PARKED", Priority: 1},
	}, nil)
	store := Store{
		Sessions:  map[string]Session{trace: {Trace: trace, Turns: fixtureTurns()}},
		Directory: dir,
	}

	res, err := store.ReadResource(DirectoryURI, trace, sessionread.DisclosureMetadata)
	if err != nil {
		t.Fatalf("ReadResource(directory): %v", err)
	}
	if len(res.Contents) != 1 {
		t.Fatalf("got %d contents, want 1", len(res.Contents))
	}
	var rows []directory.DirectoryRow
	if err := json.Unmarshal([]byte(res.Contents[0].Text), &rows); err != nil {
		t.Fatalf("unmarshal directory contents: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d directory rows, want 1 (scoped to caller's own trace): %+v", len(rows), rows)
	}
	if rows[0].TraceID != trace {
		t.Errorf("row TraceID = %q, want %q", rows[0].TraceID, trace)
	}
	if rows[0].RunState != "RUNNING" {
		t.Errorf("row RunState = %q, want RUNNING", rows[0].RunState)
	}
	if rows[0].Source != directory.SourceDrive {
		t.Errorf("row Source = %q, want %q", rows[0].Source, directory.SourceDrive)
	}

	// A different caller only ever sees its own row, never trace-A's.
	otherRes, err := store.ReadResource(DirectoryURI, "trace-other", sessionread.DisclosureMetadata)
	if err != nil {
		t.Fatalf("ReadResource(directory, trace-other): %v", err)
	}
	var otherRows []directory.DirectoryRow
	if err := json.Unmarshal([]byte(otherRes.Contents[0].Text), &otherRows); err != nil {
		t.Fatalf("unmarshal directory contents: %v", err)
	}
	if len(otherRows) != 1 || otherRows[0].TraceID != "trace-other" {
		t.Fatalf("got %+v, want just trace-other's row", otherRows)
	}
}
