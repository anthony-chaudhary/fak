package main

import (
	"io"
	"strings"
	"testing"
)

// TestDispatchPromptUsesCachedRowWithoutRefetch proves the #4167 hot-path win: when a
// routed issue's row (title/body/labels) is already in hand from the tick's list/view
// fetch and is threaded into dispatchPrompt, the prompt is built from that cached row
// WITHOUT a second `gh issue view`. The fetch seam is stubbed to fail the test if called.
func TestDispatchPromptUsesCachedRowWithoutRefetch(t *testing.T) {
	root := t.TempDir()
	oldFetchIssue := dispatchFetchIssue
	dispatchFetchIssue = func(string, int) dispatchIssueInfo {
		t.Fatalf("dispatchPrompt re-fetched the issue despite a cached body (#4167 double-fetch regression)")
		return dispatchIssueInfo{}
	}
	t.Cleanup(func() { dispatchFetchIssue = oldFetchIssue })

	cached := dispatchIssueInfo{
		Number: 4167,
		Title:  "cached routed issue",
		Body:   "This body came from the router fetch, not a second gh call.",
		Labels: []string{"dispatch"},
		State:  "OPEN",
	}
	got, err := dispatchPrompt(root, io.Discard, 4167, "cmd", cached)
	if err != nil {
		t.Fatalf("dispatchPrompt: %v", err)
	}
	if got["body"] != cached.Body {
		t.Fatalf("prompt body = %#v, want cached body %q", got["body"], cached.Body)
	}
	if prompt := dispatchMapString(got, "prompt"); !strings.Contains(prompt, "This body came from the router fetch") {
		t.Fatalf("prompt missing cached body text:\n%s", prompt)
	}
}

// TestDispatchPromptFallsBackWhenCachedBodyEmpty proves the fallback is preserved: an
// unrouted `--target-issue N` has no cached row (the picker's IssueByNumber miss yields
// a zero-value dispatchIssueInfo with an empty body), so dispatchPrompt MUST still call
// the live dispatchFetchIssue seam rather than dispatch a body-less prompt.
func TestDispatchPromptFallsBackWhenCachedBodyEmpty(t *testing.T) {
	root := t.TempDir()
	called := false
	oldFetchIssue := dispatchFetchIssue
	dispatchFetchIssue = func(_ string, issue int) dispatchIssueInfo {
		called = true
		return dispatchIssueInfo{
			Number: issue,
			Title:  "fetched fallback",
			Body:   "Fetched via gh because no cached body was threaded in.",
			Labels: []string{"dispatch"},
			State:  "OPEN",
		}
	}
	t.Cleanup(func() { dispatchFetchIssue = oldFetchIssue })

	// A zero-value cached row (Body == "") is exactly the unrouted-target cache miss.
	got, err := dispatchPrompt(root, io.Discard, 9999, "cmd", dispatchIssueInfo{})
	if err != nil {
		t.Fatalf("dispatchPrompt: %v", err)
	}
	if !called {
		t.Fatalf("dispatchPrompt did not fall back to dispatchFetchIssue on an empty cached body")
	}
	if want := "Fetched via gh because no cached body was threaded in."; got["body"] != want {
		t.Fatalf("prompt body = %#v, want fetched fallback body %q", got["body"], want)
	}
}
