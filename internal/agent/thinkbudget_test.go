package agent

import "testing"

// TestThinkBudgetUnderBudgetKeepsThinking feeds fewer reasoning tokens than the budget
// permits and asserts the counter never forces, counts every token, and stays in-span.
func TestThinkBudgetUnderBudgetKeepsThinking(t *testing.T) {
	b := NewThinkBudget(5, true) // already inside the span
	for i, tok := range []string{"let", "me", "think"} {
		if b.Observe(tok) {
			t.Fatalf("token %d (%q): forced under budget, want no force", i, tok)
		}
	}
	if b.Forced() {
		t.Fatalf("Forced() = true under budget, want false")
	}
	if b.Count() != 3 {
		t.Fatalf("Count() = %d, want 3", b.Count())
	}
	if !b.InSpan() {
		t.Fatalf("InSpan() = false, want still inside the reasoning span")
	}
}

// TestThinkBudgetBoundary pins the boundary: a budget of N permits exactly N reasoning
// tokens with no force, and the (N+1)-th token trips the force exactly once.
func TestThinkBudgetBoundary(t *testing.T) {
	const limit = 3
	b := NewThinkBudget(limit, true)
	// The first `limit` tokens are permitted.
	for i := 0; i < limit; i++ {
		if b.Observe("t") {
			t.Fatalf("token %d forced at or below budget %d, want no force", i, limit)
		}
	}
	if b.Count() != limit {
		t.Fatalf("Count() = %d, want %d at the boundary", b.Count(), limit)
	}
	// The next token spends the budget and forces exactly once.
	if !b.Observe("t") {
		t.Fatalf("token %d did not force, want force at budget+1", limit)
	}
	if !b.Forced() {
		t.Fatalf("Forced() = false after the force, want true")
	}
	// The forcing token is NOT counted — it is replaced by the reasoning-end marker.
	if b.Count() != limit {
		t.Fatalf("Count() = %d after force, want %d (forcing token uncounted)", b.Count(), limit)
	}
}

// TestThinkBudgetForceRaisedExactlyOnce checks the one-way latch: after the force fires,
// no later token re-raises it, and the span is reported closed.
func TestThinkBudgetForceRaisedExactlyOnce(t *testing.T) {
	b := NewThinkBudget(1, true)
	if b.Observe("a") { // first reasoning token permitted (count 0 < 1)
		t.Fatalf("first token forced, want permitted under budget 1")
	}
	if !b.Observe("b") { // second token spends the budget
		t.Fatalf("second token did not force, want force")
	}
	raises := 0
	for _, tok := range []string{"c", "d", "e"} {
		if b.Observe(tok) {
			raises++
		}
	}
	if raises != 0 {
		t.Fatalf("force re-raised %d times after the latch, want 0", raises)
	}
	if b.InSpan() {
		t.Fatalf("InSpan() = true after force, want span reported closed")
	}
}

// TestThinkBudgetUnlimited asserts a negative budget never forces, however long the
// reasoning span runs.
func TestThinkBudgetUnlimited(t *testing.T) {
	b := NewThinkBudget(-1, true)
	for i := 0; i < 100; i++ {
		if b.Observe("t") {
			t.Fatalf("token %d forced under an unlimited budget, want never", i)
		}
	}
	if b.Forced() {
		t.Fatalf("Forced() = true under unlimited budget, want false")
	}
	if b.Count() != 100 {
		t.Fatalf("Count() = %d, want 100", b.Count())
	}
}

// TestThinkBudgetZeroForcesImmediately asserts a zero budget forbids any reasoning
// token: the very first reasoning token trips the force.
func TestThinkBudgetZeroForcesImmediately(t *testing.T) {
	b := NewThinkBudget(0, true)
	if !b.Observe("t") {
		t.Fatalf("zero budget did not force on the first reasoning token, want immediate force")
	}
	if b.Count() != 0 {
		t.Fatalf("Count() = %d under zero budget, want 0", b.Count())
	}
	if !b.Forced() {
		t.Fatalf("Forced() = false, want true after immediate force")
	}
}

// TestThinkBudgetOutsideSpanDoesNotCount asserts tokens before the open marker and at or
// after the close marker do not count against the budget, and that a natural close never
// forces.
func TestThinkBudgetOutsideSpanDoesNotCount(t *testing.T) {
	b := NewThinkBudget(2, false) // not pre-seeded; must see the open marker first
	// Pre-span tokens must not count.
	for _, tok := range []string{"prompt", "prefix"} {
		if b.Observe(tok) {
			t.Fatalf("pre-span token %q forced, want no force", tok)
		}
	}
	if b.Count() != 0 {
		t.Fatalf("Count() = %d before the span opened, want 0", b.Count())
	}
	// Open marker, then two reasoning tokens (exactly the budget), then a natural close.
	if b.Observe(thinkOpen) {
		t.Fatalf("open marker forced, want no force")
	}
	for i, tok := range []string{"one", "two"} {
		if b.Observe(tok) {
			t.Fatalf("reasoning token %d (%q) forced at budget 2, want no force", i, tok)
		}
	}
	if b.Observe(thinkClose) {
		t.Fatalf("natural close marker forced, want no force")
	}
	// Post-span answer tokens must not count or force.
	for _, tok := range []string{"the", "answer"} {
		if b.Observe(tok) {
			t.Fatalf("post-span token %q forced, want no force", tok)
		}
	}
	if b.Count() != 2 {
		t.Fatalf("Count() = %d, want exactly 2 reasoning tokens", b.Count())
	}
	if b.Forced() {
		t.Fatalf("Forced() = true after a natural close, want false")
	}
}

// TestThinkBudgetSplitMarker asserts an open marker split across two tokens still opens
// the span so the following tokens count.
func TestThinkBudgetSplitMarker(t *testing.T) {
	b := NewThinkBudget(1, false)
	// "<think>" arrives split as "<thi" + "nk>".
	if b.Observe("<thi") || b.Observe("nk>") {
		t.Fatalf("split open marker forced, want no force")
	}
	if !b.InSpan() {
		t.Fatalf("InSpan() = false after a split open marker, want span opened")
	}
	if b.Observe("first") { // count 0 < 1, permitted
		t.Fatalf("first reasoning token forced, want permitted")
	}
	if !b.Observe("second") { // spends the budget
		t.Fatalf("second reasoning token did not force, want force at budget+1")
	}
}

// TestForceIndex exercises the whole-stream helper for the forced case and the
// never-forced (natural close) case.
func TestForceIndex(t *testing.T) {
	// Budget 2, pre-seeded in-span: tokens 0,1 permitted, token 2 forces.
	if got := ForceIndex([]string{"a", "b", "c", "d"}, 2, true); got != 2 {
		t.Fatalf("ForceIndex forced case = %d, want 2", got)
	}
	// Natural close before the budget is spent → never forces.
	toks := []string{thinkOpen, "a", thinkClose, "answer"}
	if got := ForceIndex(toks, 5, false); got != -1 {
		t.Fatalf("ForceIndex natural-close case = %d, want -1", got)
	}
}
