package session

// compactregrowth_replay_test.go — failure-class proof for the #5254 counterfactual dedup replay.
//
// The replay's job is to bound what a candidate repeated-tool-result mechanism WOULD have
// collapsed over windows that already exist, because the DoD's own witness ("materially reduced
// on new sessions") is defined over sessions that do not exist yet. A bound is only worth
// anything if it can also come out NEGATIVE, so these fixtures pin the three ways it must:
//
//   - a within-window repeat is credited (the win),
//   - a cross-fire repeat is NOT credited (reach — the copy was compacted out of the wire, so no
//     within-wire fold can match it), and
//   - a span that was never a duplicate must survive byte-identical (loss — the false-positive
//     guard, asserted against a deliberately over-eager folder so the guard is shown to be able
//     to fail).

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// replayBody builds a multi-line tool-result body whose row clears RegrowthDupMinBytes (so a
// repeat registers as duplication at all) and whose line count clears the 8-line run floor a
// line-matching folder needs.
func replayBody(tag string) string {
	var b strings.Builder
	for i := 0; b.Len() < 3000; i++ {
		fmt.Fprintf(&b, "%s line %03d: the quick brown fox jumps over the lazy dog and keeps going\n", tag, i)
	}
	return b.String()
}

// replayRollout renders a Codex-shaped rollout: an optional pre-fire tool-result run, a
// compaction fire, then the post-fire run. Bodies are given as plain text and JSON-encoded here,
// exactly as Codex writes function_call_output.output.
func replayRollout(t *testing.T, preFire, postFire []string) string {
	t.Helper()
	var b strings.Builder
	row := func(ts, typ string, payload any) {
		p, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal row: %v", err)
		}
		fmt.Fprintf(&b, `{"timestamp":%q,"type":%q,"payload":%s}`+"\n", ts, typ, p)
	}
	tokens := func(ts string, last int) {
		row(ts, "event_msg", map[string]any{
			"type": "token_count",
			"info": map[string]any{
				"total_token_usage":    map[string]any{"input_tokens": 400000},
				"last_token_usage":     map[string]any{"input_tokens": last, "cached_input_tokens": 0},
				"model_context_window": 258400,
			},
		})
	}
	toolPair := func(ts, id, body string) {
		row(ts, "response_item", map[string]any{
			"type": "function_call", "id": id, "name": "shell",
			"arguments": `{"command":"` + id + `"}`, "call_id": id,
		})
		row(ts, "response_item", map[string]any{
			"type": "function_call_output", "call_id": id, "output": body,
		})
	}

	row("2026-07-01T10:00:00.000Z", "session_meta", map[string]any{"id": "regrow-replay", "cwd": "/w"})
	tokens("2026-07-01T10:05:00.000Z", 235000)
	// Pre- and post-fire call ids are the SAME WIDTH on purpose. The audit keys duplicates on
	// {content hash, ROW length}, and the row length includes the call id, so ids of differing
	// width would stop two byte-identical outputs from matching at all — see
	// TestRegrowthReplayDupKeyIsRowLengthSensitive, which pins that as a real property of the
	// audit rather than letting it silently weaken this fixture.
	for i, body := range preFire {
		toolPair(fmt.Sprintf("2026-07-01T10:0%d:00.000Z", 6+i), fmt.Sprintf("call_pre_%02d", i), body)
	}
	row("2026-07-01T10:10:00.000Z", "compacted", map[string]any{"message": ""})
	tokens("2026-07-01T10:10:30.000Z", 25000)
	for i, body := range postFire {
		toolPair(fmt.Sprintf("2026-07-01T10:%02d:00.000Z", 11+i), fmt.Sprintf("call_pst_%02d", i), body)
		tokens(fmt.Sprintf("2026-07-01T10:%02d:30.000Z", 11+i), 30000+10000*i)
	}
	return b.String()
}

func runReplay(t *testing.T, rollout string, opt RegrowthReplayOptions) RegrowthReplayStat {
	t.Helper()
	_, st, err := ScanCompactRolloutReplay(strings.NewReader(rollout), "mem", int64(len(rollout)), opt)
	if err != nil {
		t.Fatalf("replay scan: %v", err)
	}
	return st
}

// keepEarliestFold is a stand-in for the shipped mechanism's SEMANTICS (agent.ElideMessages'
// cross-turn fold): the earliest occurrence of a body stays verbatim and every later byte-identical
// occurrence collapses to a one-line in-band pointer. internal/session is tier 1 and cannot import
// the real elider (internal/agent, tier 4); the end-to-end join against the real one lives in
// cmd/fak. What this file proves is that the SCORER is sound — that it credits, withholds, and
// detects loss correctly — which is what makes the cmd/fak number believable.
func keepEarliestFold(bodies []string) []string {
	out := make([]string, len(bodies))
	first := make(map[string]int, len(bodies))
	for i, b := range bodies {
		if j, ok := first[b]; ok {
			out[i] = fmt.Sprintf("[fak dedup: identical to output shown earlier, turn %d]\n", j)
			continue
		}
		first[b] = i
		out[i] = b
	}
	return out
}

// A body repeated three times under DISTINCT call ids inside one post-fire window is the issue's
// headline shape. The replay must credit it: the anomaly fires before, does not fire after, and
// the collapsed bytes are counted as in-window reach rather than cross-fire.
func TestRegrowthReplayCollapsesInWindowRepeats(t *testing.T) {
	body := replayBody("repeat")
	rollout := replayRollout(t, nil, []string{body, body, body})
	st := runReplay(t, rollout, RegrowthReplayOptions{Fold: keepEarliestFold, MinDupLines: 8})

	if st.Windows != 1 || st.ToolResultRows != 3 {
		t.Fatalf("windows=%d rows=%d, want 1 window of 3 tool-result rows", st.Windows, st.ToolResultRows)
	}
	if st.AnomalyWindowsBefore != 1 {
		t.Fatalf("anomaly_windows_before=%d, want 1 — the fixture does not reproduce the defect it measures", st.AnomalyWindowsBefore)
	}
	if st.DupRowsBefore != 2 {
		t.Errorf("dup_rows_before=%d, want 2 (the 2nd and 3rd copies)", st.DupRowsBefore)
	}
	if st.InWindowDupRows != 2 || st.CrossFireDupRows != 0 {
		t.Errorf("reach split = %d in-window / %d cross-fire, want 2/0 — both repeats are inside the wire", st.InWindowDupRows, st.CrossFireDupRows)
	}
	if st.AnomalyWindowsAfter != 0 || st.WindowsCollapsed != 1 {
		t.Errorf("after=%d collapsed=%d, want 0/1 — the folded window must stop carrying %s",
			st.AnomalyWindowsAfter, st.WindowsCollapsed, AnomalyRepeatedToolResult)
	}
	if st.DupBytesAfter >= st.DupBytesBefore {
		t.Errorf("dup_bytes %d -> %d, want a strict reduction", st.DupBytesBefore, st.DupBytesAfter)
	}
	if st.ShedBytes <= 0 || st.FoldedRows != 2 {
		t.Errorf("shed=%d folded_rows=%d, want >0 bytes over exactly the 2 later copies", st.ShedBytes, st.FoldedRows)
	}
	// Lossless-by-relocation: every removed line is still reachable in an earlier body, none lost.
	if st.RemovedLinesLost != 0 {
		t.Errorf("removed_lines_lost=%d, want 0 — folded bytes must stay reachable in-band", st.RemovedLinesLost)
	}
	if st.RemovedLinesRelocated <= 0 {
		t.Errorf("removed_lines_relocated=%d, want >0 — the fold did remove lines, and they must be accounted", st.RemovedLinesRelocated)
	}
	if st.WholeBodyDupRows != 2 || st.PartialFoldRows != 0 {
		t.Errorf("whole-body dup folds=%d partial=%d, want 2/0", st.WholeBodyDupRows, st.PartialFoldRows)
	}
}

// REACH. The audit's duplicate table is session-wide and spans fires, so a body first seen BEFORE
// the fire and re-appended after it is correctly typed a duplicate — but its earlier copy was
// compacted out of the wire, leaving a within-wire fold nothing to match against. Those bytes must
// be withheld from the claimable bound. This is the issue's own "the same output re-entering the
// window it was just compacted out of", and it is exactly the part the shipped mechanism cannot fix.
func TestRegrowthReplayWithholdsCrossFireRepeats(t *testing.T) {
	body := replayBody("crossfire")
	rollout := replayRollout(t, []string{body}, []string{body, replayBody("other")})
	st := runReplay(t, rollout, RegrowthReplayOptions{Fold: keepEarliestFold, MinDupLines: 8})

	if st.DupRowsBefore != 1 {
		t.Fatalf("dup_rows_before=%d, want 1 — the post-fire copy must still be typed a duplicate", st.DupRowsBefore)
	}
	if st.CrossFireDupRows != 1 || st.InWindowDupRows != 0 {
		t.Fatalf("reach split = %d in-window / %d cross-fire, want 0/1 — the partner copy is on the far side of the fire",
			st.InWindowDupRows, st.CrossFireDupRows)
	}
	if st.CrossFireDupBytes <= 0 {
		t.Errorf("cross_fire_dup_bytes=%d, want the withheld bytes counted, not dropped", st.CrossFireDupBytes)
	}
	if st.ShedBytes != 0 || st.FoldedRows != 0 {
		t.Errorf("shed=%d folded=%d, want 0/0 — a within-wire fold cannot collapse a cross-fire repeat", st.ShedBytes, st.FoldedRows)
	}
	if st.WindowsCollapsed != 0 {
		t.Errorf("windows_collapsed=%d, want 0 — crediting this window would overstate the bound", st.WindowsCollapsed)
	}
}

// LOSS, the guard that actually matters: a span the fold removes must still be reachable verbatim
// in an earlier body. Asserted twice — an honest fold destroys nothing, and a deliberately
// over-eager one is CAUGHT. A guard that cannot fail proves nothing.
func TestRegrowthReplayDetectsDestroyedSpans(t *testing.T) {
	distinct := []string{replayBody("alpha"), replayBody("bravo"), replayBody("charlie")}
	rollout := replayRollout(t, nil, distinct)

	honest := runReplay(t, rollout, RegrowthReplayOptions{Fold: keepEarliestFold, MinDupLines: 8})
	if honest.ToolResultRows != 3 {
		t.Fatalf("rows=%d, want 3", honest.ToolResultRows)
	}
	if honest.RemovedLinesLost != 0 || honest.RemovedLinesRelocated != 0 {
		t.Errorf("lost=%d relocated=%d, want 0/0 — nothing repeats here, so nothing may be removed",
			honest.RemovedLinesLost, honest.RemovedLinesRelocated)
	}
	if honest.ShedBytes != 0 || honest.AnomalyWindowsBefore != 0 {
		t.Errorf("shed=%d anomaly_before=%d, want 0/0 — nothing here is duplicated", honest.ShedBytes, honest.AnomalyWindowsBefore)
	}

	// An over-eager mechanism that rewrites every span regardless of repetition is the exact
	// failure a "bytes saved" headline would otherwise launder as a win: it sheds the most bytes
	// of any folder here AND destroys content that exists nowhere else.
	greedy := func(bodies []string) []string {
		out := make([]string, len(bodies))
		for i := range bodies {
			out[i] = "[dropped]\n"
		}
		return out
	}
	caught := runReplay(t, rollout, RegrowthReplayOptions{Fold: greedy, MinDupLines: 8})
	if caught.RemovedLinesLost <= 0 {
		t.Errorf("removed_lines_lost=%d, want >0 — the loss check must catch a folder that destroys unrepeated content", caught.RemovedLinesLost)
	}
	if caught.ShedBytes <= caught.RemovedLinesLost {
		t.Logf("greedy shed=%d lost lines=%d", caught.ShedBytes, caught.RemovedLinesLost)
	}
	if caught.PartialFoldRows != 3 {
		t.Errorf("partial_fold_rows=%d, want 3 — every body was rewritten though none was a whole-body duplicate", caught.PartialFoldRows)
	}
}

// A span-level fold legitimately collapses a line run SHARED by two otherwise-different bodies —
// two runs of the same command whose outputs differ only in a trailing line. The audit's body-level
// duplicate accounting cannot see that at all (neither body is byte-identical to the other), so
// this is reach the REPEATED_TOOL_RESULT headline structurally undercounts. It must be scored as a
// relocation, never as loss.
func TestRegrowthReplayCreditsSharedSpanAcrossDistinctBodies(t *testing.T) {
	shared := replayBody("shared")
	rollout := replayRollout(t, nil, []string{shared + "tail A\n", shared + "tail B\n"})
	st := runReplay(t, rollout, RegrowthReplayOptions{Fold: sharedRunFold, MinDupLines: 8})

	if st.DupRowsBefore != 0 {
		t.Fatalf("dup_rows_before=%d, want 0 — the two bodies are NOT byte-identical, so the audit sees no duplication", st.DupRowsBefore)
	}
	if st.RemovedLinesLost != 0 {
		t.Errorf("removed_lines_lost=%d, want 0 — the shared run is still reachable in the first body", st.RemovedLinesLost)
	}
	if st.RemovedLinesRelocated <= 0 || st.PartialFoldRows != 1 {
		t.Errorf("relocated=%d partial=%d, want >0 / 1 — the shared prefix must fold in the SECOND body only",
			st.RemovedLinesRelocated, st.PartialFoldRows)
	}
	if st.ShedBytes <= 0 {
		t.Errorf("shed=%d, want >0 — this is real saving the body-level dup accounting never counted", st.ShedBytes)
	}
}

// sharedRunFold collapses the longest common LINE PREFIX a body shares with any strictly-earlier
// body, which is the span-level shape of the shipped cross-turn matcher.
func sharedRunFold(bodies []string) []string {
	out := make([]string, len(bodies))
	copy(out, bodies)
	for i := 1; i < len(bodies); i++ {
		lines := strings.SplitAfter(bodies[i], "\n")
		best := 0
		for j := 0; j < i; j++ {
			prev := strings.SplitAfter(bodies[j], "\n")
			n := 0
			for n < len(lines) && n < len(prev) && lines[n] == prev[n] {
				n++
			}
			if n > best {
				best = n
			}
		}
		if best >= 8 {
			out[i] = "[fak dedup: " + fmt.Sprint(best) + " lines identical to output shown earlier]\n" +
				strings.Join(lines[best:], "")
		}
	}
	return out
}

// A mechanism that changes the span COUNT is not comparable on this axis; scoring it would credit
// a deletion as a fold. It must be refused and counted, never silently accepted.
func TestRegrowthReplayRefusesShapeChangingFolder(t *testing.T) {
	body := replayBody("repeat")
	rollout := replayRollout(t, nil, []string{body, body})
	st := runReplay(t, rollout, RegrowthReplayOptions{
		Fold: func(bodies []string) []string { return bodies[:len(bodies)-1] },
	})
	if st.ShapeErrorWindows != 1 {
		t.Errorf("shape_error_windows=%d, want 1", st.ShapeErrorWindows)
	}
	if st.WindowsCollapsed != 0 || st.ShedBytes != 0 {
		t.Errorf("collapsed=%d shed=%d, want 0/0 — a dropped span must never score as a saving", st.WindowsCollapsed, st.ShedBytes)
	}
}

// The audit keys a duplicate on {content hash, ROW length}, and the row length includes the
// per-call id — so two byte-identical tool outputs whose call ids differ in WIDTH do not match,
// and the window never raises REPEATED_TOOL_RESULT for them. Pinned here because it means the
// published 296-window / 12.5 MB figure is a LOWER bound on repetition, and therefore so is every
// bound this replay derives from it. Codex call ids are effectively fixed-width in practice, which
// is why the headline still stands; this test exists so the assumption is visible instead of
// inherited.
func TestRegrowthReplayDupKeyIsRowLengthSensitive(t *testing.T) {
	body := replayBody("samebody")
	var b strings.Builder
	b.WriteString(`{"timestamp":"2026-07-01T10:00:00.000Z","type":"session_meta","payload":{"id":"k","cwd":"/w"}}` + "\n")
	b.WriteString(`{"timestamp":"2026-07-01T10:10:00.000Z","type":"compacted","payload":{"message":""}}` + "\n")
	enc, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	// Same output, same tool, call ids of DIFFERENT width.
	for _, id := range []string{"call_x", "call_xxxxxx"} {
		fmt.Fprintf(&b, `{"timestamp":"2026-07-01T10:11:00.000Z","type":"response_item","payload":{"type":"function_call","id":%q,"name":"shell","arguments":"{}","call_id":%q}}`+"\n", id, id)
		fmt.Fprintf(&b, `{"timestamp":"2026-07-01T10:11:10.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":%q,"output":%s}}`+"\n", id, enc)
	}
	st := runReplay(t, b.String(), RegrowthReplayOptions{Fold: keepEarliestFold, MinDupLines: 8})

	if st.ToolResultRows != 2 {
		t.Fatalf("tool_result_rows=%d, want 2", st.ToolResultRows)
	}
	if st.DupRowsBefore != 0 {
		t.Fatalf("dup_rows_before=%d, want 0 — this test documents the row-length sensitivity; if the audit's key stops including row length, update the note above", st.DupRowsBefore)
	}
	// The bodies ARE byte-identical, so the fold still collapses the second one. The replay counts
	// that shed even though the audit never typed the pair a duplicate — which is exactly why the
	// claimable bound is reported against the audit's own dup accounting and not against ShedBytes.
	if st.ShedBytes <= 0 {
		t.Errorf("shed=%d, want >0 — the content is identical even though the audit's key missed it", st.ShedBytes)
	}
}

// The replay is opt-in: the production scan is unarmed, unchanged, and still body-blind. The
// bodies the replay holds in memory must never reach the report the audit serialises.
func TestRegrowthReplayOffByDefaultAndNeverEmitsBodies(t *testing.T) {
	body := replayBody("MUST_NOT_LEAK")
	rollout := replayRollout(t, nil, []string{body, body})

	rep, err := ScanCompactRollout(strings.NewReader(rollout), "mem", int64(len(rollout)))
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	armed, st, err := ScanCompactRolloutReplay(strings.NewReader(rollout), "mem", int64(len(rollout)), RegrowthReplayOptions{Fold: keepEarliestFold})
	if err != nil {
		t.Fatalf("armed scan: %v", err)
	}
	if st.Rollouts != 1 || st.Windows != 1 {
		t.Fatalf("stat = %+v, want one rollout / one window", st)
	}
	// Arming the replay must not perturb the audit it is measuring.
	want, _ := json.Marshal(rep)
	got, _ := json.Marshal(armed)
	if string(want) != string(got) {
		t.Errorf("arming the replay changed the audit report; it must be observation-only")
	}
	if _, _, err := ScanCompactRolloutReplay(strings.NewReader(rollout), "mem", int64(len(rollout)), RegrowthReplayOptions{}); err != nil {
		t.Fatalf("zero-value options must be identity: %v", err)
	}
	for _, blob := range [][]byte{want, got} {
		if strings.Contains(string(blob), "MUST_NOT_LEAK") {
			t.Fatal("a tool-result body reached the serialised report")
		}
	}
	if b, _ := json.Marshal(st); strings.Contains(string(b), "MUST_NOT_LEAK") {
		t.Fatal("a tool-result body reached the replay stat; it must carry counts only")
	}
}
