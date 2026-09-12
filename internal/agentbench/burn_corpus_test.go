package agentbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestBurnInCorpusUsesFourDisjointCausalAreas(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	addBurnCorpusMaterial(t, repo)
	normal, err := buildNormalCorpus(context.Background(), &normalCorpusEncoder{}, repo)
	if err != nil {
		t.Fatal(err)
	}
	encoder := &burnMaterialEncoder{}
	corpus, err := buildBurnInCorpus(context.Background(), encoder, normal, repo)
	if err != nil {
		t.Fatal(err)
	}
	if encoder.finalCalls != 8*32 || encoder.measurementCalls == 0 {
		t.Fatalf("reference encodes final/area-fit=%d/%d want 256 plus explicit stable-template area measurements", encoder.finalCalls, encoder.measurementCalls)
	}
	if len(corpus.Areas) != 4 {
		t.Fatalf("source areas=%d want four pinned disjoint areas", len(corpus.Areas))
	}
	type span struct {
		path       string
		start, end int
	}
	var spans []span
	areaDigests := map[string]bool{}
	for name, area := range corpus.Areas {
		if area.SHA256 == "" || area.content == "" || len(area.Provenance) == 0 || area.ReferenceTokens < 7000 || area.ReferenceTokens > 9000 {
			t.Fatalf("area %s lacks pinned source material: %+v", name, area)
		}
		if areaDigests[area.SHA256] {
			t.Fatalf("area %s duplicates source digest %s", name, area.SHA256)
		}
		areaDigests[area.SHA256] = true
		for _, provenance := range area.Provenance {
			if provenance.Revision != normal.Revision || provenance.Path == "" || provenance.StartLine <= 0 || provenance.EndLine < provenance.StartLine {
				t.Fatalf("area %s has unpinned provenance: %+v", name, provenance)
			}
			for _, prior := range spans {
				if prior.path == provenance.Path && provenance.StartLine <= prior.end && prior.start <= provenance.EndLine {
					t.Fatalf("overlapping area provenance in %s: %d-%d overlaps %d-%d", provenance.Path, provenance.StartLine, provenance.EndLine, prior.start, prior.end)
				}
			}
			spans = append(spans, span{provenance.Path, provenance.StartLine, provenance.EndLine})
		}
	}
	for _, session := range corpus.Sessions {
		if len(session.Turns) != 32 || len(session.AreaTransitions) != 4 {
			t.Fatalf("session %s turns/transitions=%d/%d want 32/4", session.ID, len(session.Turns), len(session.AreaTransitions))
		}
		for i, want := range []string{"A", "B", "C", "D"} {
			transition := session.AreaTransitions[i]
			if transition.Area != want || transition.BeforeSHA256 == "" || transition.AfterSHA256 == "" || transition.BeforeSHA256 == transition.AfterSHA256 {
				t.Fatalf("session %s transition %d not bound to changed request material: %+v", session.ID, i, transition)
			}
		}
		rotationTurns := map[int]bool{}
		for _, rotation := range session.HistoryRotations {
			if rotation.Turn <= 1 || rotation.RemovedSHA256 == "" || rotation.AddedSHA256 == "" || rotation.BeforeSHA256 == rotation.AfterSHA256 || rotation.CandidatePromptTokens+rotation.ReservedOutputTokens <= 32768 {
				t.Fatalf("session %s has unbound history rotation: %+v", session.ID, rotation)
			}
			rotationTurns[rotation.Turn] = true
		}
		if len(rotationTurns) == 0 {
			t.Fatalf("session %s never rotates an old recorded assistant/tool pair", session.ID)
		}
		if first := session.Turns[0].PromptTokens; first < 13000 || first > 15000 {
			t.Fatalf("session %s first prompt=%d want exact-reference 13K-15K", session.ID, first)
		}
		peak := 0
		caps := map[int]int{}
		for i := 1; i < len(session.Turns); i++ {
			prev, next := session.Turns[i-1], session.Turns[i]
			causalAppend := messageHistoryPrefix(prev.Request.Messages, next.Request.Messages) && len(next.Request.Messages) > len(prev.Request.Messages)
			causalRotation := rotationTurns[i+1] && next.HistoryTokens <= prev.HistoryTokens
			if (!causalAppend && !causalRotation) || next.Encoding.RenderedSHA256 == prev.Encoding.RenderedSHA256 {
				t.Fatalf("session %s turn %d is not causally chained recorded history", session.ID, i+1)
			}
			if next.PromptTokens+next.OutputTokens > 32768 {
				t.Fatalf("session %s turn %d exceeds native envelope", session.ID, i+1)
			}
			peak = max(peak, next.PromptTokens)
		}
		for i, turn := range session.Turns {
			caps[turn.OutputTokens]++
			area := []string{"A", "B", "C", "D"}[i/8]
			if !requestContainsMaterial(turn.Request, corpus.Areas[area].content) {
				t.Fatalf("session %s turn %d lacks actual area %s bytes", session.ID, i+1, area)
			}
		}
		if peak < 28000 || peak > 31744 {
			t.Fatalf("session %s peak prompt=%d want 28K-31744", session.ID, peak)
		}
		if caps[128] != 24 || caps[256] != 6 || caps[1024] != 2 {
			t.Fatalf("session %s output-cap mix=%v want repeated 8x128/3x256/1x1024", session.ID, caps)
		}
	}
}

type burnMaterialEncoder struct{ finalCalls, measurementCalls int }

func (e *burnMaterialEncoder) Encode(_ context.Context, req referenceRequest) (referenceEncoding, error) {
	raw, _ := json.Marshal(req)
	if req.OutputTokens == 0 {
		e.measurementCalls++
	} else {
		e.finalCalls++
	}
	if req.OutputTokens != 0 && len(raw) < 12000 {
		return referenceEncoding{}, fmt.Errorf("request lacks full source/history material: %d bytes", len(raw))
	}
	tokens := len(raw) / 2
	if tokens+req.OutputTokens > 32768 {
		return referenceEncoding{}, fmt.Errorf("measured request exceeds context: %d+%d", tokens, req.OutputTokens)
	}
	sum := sha256.Sum256(raw)
	return referenceEncoding{ModelID: "fixture-model", RendererID: "renderer", TokenizerID: "tokenizer", TokenIDs: make([]int, tokens), PromptTokens: tokens, ContextWindowTokens: 32768, ReservedOutputTokens: req.OutputTokens, RenderedSHA256: fmt.Sprintf("%x", sum)}, nil
}

func addBurnCorpusMaterial(t *testing.T, repo string) {
	t.Helper()
	for area := 0; area < 4; area++ {
		var source strings.Builder
		source.WriteString(fmt.Sprintf("package area%d\n\n", area))
		for line := 0; line < 500; line++ {
			fmt.Fprintf(&source, "// pinned-area-%d-line-%04d unique committed benchmark source material\n", area, line)
		}
		if err := os.WriteFile(filepath.Join(repo, fmt.Sprintf("burn_area_%d.go", area)), []byte(source.String()), 0600); err != nil {
			t.Fatal(err)
		}
	}
	cmd := exec.Command("git", "add", ".")
	cmd.Dir = repo
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git add burn material: %v: %s", err, out)
	}
	cmd = exec.Command("git", "commit", "-q", "-m", "burn corpus material")
	cmd.Dir = repo
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git commit burn material: %v: %s", err, out)
	}
}

func messageHistoryPrefix(prior, next []chatMessage) bool {
	if len(next) < len(prior) {
		return false
	}
	for i := range prior {
		a, _ := json.Marshal(prior[i])
		b, _ := json.Marshal(next[i])
		if !bytes.Equal(a, b) {
			return false
		}
	}
	return true
}

func requestContainsMaterial(request referenceRequest, material string) bool {
	for _, message := range request.Messages {
		if strings.Contains(message.Content, material) {
			return true
		}
	}
	return false
}
