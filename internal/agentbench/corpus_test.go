package agentbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestNormalCorpusExactReferenceEnvelope(t *testing.T) {
	repo := normalCorpusRepo(t, true)
	encoder := &normalCorpusEncoder{}
	corpus, err := buildNormalCorpus(context.Background(), encoder, repo)
	if err != nil {
		t.Fatal(err)
	}
	if corpus.Schema == "" || corpus.Revision == "" || corpus.Revision == "HEAD" {
		t.Fatalf("corpus identity = schema %q revision %q", corpus.Schema, corpus.Revision)
	}
	assertCorpusLayerBand(t, corpus.SystemTools, 3500, 4500)
	assertCorpusLayerBand(t, corpus.RepositoryMap, 1500, 2500)
	for name, layer := range corpus.Areas {
		if name != "A" && name != "B" {
			t.Fatalf("unexpected area %q", name)
		}
		assertCorpusLayerBand(t, layer, 7000, 9000)
	}
	areaCounts := map[string]int{}
	seenSessions := map[string]bool{}
	for _, session := range corpus.Sessions {
		if session.ID == "" || seenSessions[session.ID] || len(session.Turns) != 12 {
			t.Fatalf("invalid session identity/turn count: %+v", session)
		}
		seenSessions[session.ID] = true
		areaCounts[session.Area]++
		caps := map[int]int{}
		for i, turn := range session.Turns {
			if turn.Ordinal != i+1 || turn.PromptTokens != len(turn.Encoding.TokenIDs) || turn.HistoryTokens < 0 {
				t.Fatalf("session %s turn %d accounting = %+v", session.ID, i+1, turn)
			}
			caps[turn.OutputTokens]++
			if turn.Encoding.ReservedOutputTokens != turn.OutputTokens || turn.PromptTokens+turn.OutputTokens > 32768 {
				t.Fatalf("session %s turn %d exceeds exact envelope: %+v", session.ID, i+1, turn)
			}
			for _, digest := range turn.LayerDigests {
				if len(digest) != 64 {
					t.Fatalf("invalid layer digest %q", digest)
				}
			}
		}
		if caps[128] != 8 || caps[256] != 3 || caps[1024] != 1 || session.Turns[0].PromptTokens < 13000 || session.Turns[0].PromptTokens > 15000 || session.Turns[11].PromptTokens < 28000 || session.Turns[11].PromptTokens > 31744 {
			t.Fatalf("session %s caps/envelope = %#v first=%d last=%d", session.ID, caps, session.Turns[0].PromptTokens, session.Turns[11].PromptTokens)
		}
	}
	if areaCounts["A"] != 6 || areaCounts["B"] != 2 {
		t.Fatalf("area session allocation = %#v, want 6A/2B", areaCounts)
	}
	if encoder.EstimatedFromCharacters() {
		t.Fatal("corpus used a characters/4 token estimate instead of reference Encode responses")
	}

	allowedKinds := map[string]bool{"committed_source": true, "tool_contract": true, "search_record": true, "diff_record": true, "test_record": true}
	seenProvenance := map[string]bool{}
	for _, layer := range append([]corpusLayer{corpus.SystemTools, corpus.RepositoryMap}, corpus.Areas["A"], corpus.Areas["B"]) {
		if layer.SHA256 == "" || layer.Bytes <= 0 || len(layer.Provenance) == 0 {
			t.Fatalf("layer lacks immutable material/provenance: %+v", layer)
		}
		for _, provenance := range layer.Provenance {
			key := fmt.Sprintf("%s:%s:%d:%d:%s", provenance.Kind, provenance.Path, provenance.StartLine, provenance.EndLine, provenance.SHA256)
			if !allowedKinds[provenance.Kind] || provenance.Revision == "" || provenance.SHA256 == "" || seenProvenance[key] {
				t.Fatalf("invalid or duplicate provenance: %+v", provenance)
			}
			seenProvenance[key] = true
		}
	}

	var commonTokens, commonOffset int
	var commonDigest string
	nonces := map[string]bool{}
	for _, concurrency := range []int{1, 2, 4, 8} {
		precondition, ok := corpus.Preconditions[concurrency]
		if !ok || precondition.Concurrency != concurrency || precondition.Nonce == "" || nonces[precondition.Nonce] {
			t.Fatalf("C%d precondition = %+v present=%v", concurrency, precondition, ok)
		}
		nonces[precondition.Nonce] = true
		if commonTokens == 0 {
			commonTokens, commonOffset, commonDigest = precondition.PromptTokens, precondition.NonceTokenOffset, precondition.EncodingSHA256
		} else if precondition.PromptTokens != commonTokens || precondition.NonceTokenOffset != commonOffset || precondition.EncodingSHA256 != commonDigest {
			t.Fatalf("C%d precondition is not equivalent apart from nonce: %+v", concurrency, precondition)
		}
	}
	first, second := corpus.FrozenInputs(), corpus.FrozenInputs()
	first.area["A"] = "mutated"
	if reflect.DeepEqual(first.area, second.area) || second.area["A"] == "mutated" {
		t.Fatal("FrozenInputs returned aliased mutable area map")
	}

	if _, err := buildNormalCorpus(context.Background(), &normalCorpusEncoder{undersize: true}, normalCorpusRepo(t, false)); err == nil {
		t.Fatal("insufficient source/reference count bands produced a corpus")
	}
}

func assertCorpusLayerBand(t *testing.T, layer corpusLayer, low, high int) {
	t.Helper()
	if layer.ReferenceTokens < low || layer.ReferenceTokens > high {
		t.Fatalf("layer %s reference tokens = %d, want %d..%d", layer.Name, layer.ReferenceTokens, low, high)
	}
}

type normalCorpusEncoder struct {
	mu        sync.Mutex
	calls     [][]byte
	undersize bool
}

func (e *normalCorpusEncoder) Encode(_ context.Context, req referenceRequest) (referenceEncoding, error) {
	body, _ := json.Marshal(req)
	e.mu.Lock()
	e.calls = append(e.calls, append([]byte(nil), body...))
	e.mu.Unlock()
	text := string(body)
	tokens := 14000
	switch {
	case req.OutputTokens == 0 && strings.Contains(strings.ToLower(text), "tool"):
		tokens = 4000
	case req.OutputTokens == 0 && strings.Contains(strings.ToLower(text), "repository"):
		tokens = 2000
	case req.OutputTokens == 0:
		tokens = 8000
	default:
		turns := strings.Count(text, `"role"`)
		tokens = 14000 + (turns-1)*1400
		if tokens > 31744 {
			tokens = 31744
		}
	}
	if e.undersize {
		tokens = 10
	}
	ids := make([]int, tokens)
	for i := range ids {
		ids[i] = i % 251
	}
	sum := sha256.Sum256(body)
	return referenceEncoding{ModelID: "fixture-model", RendererID: "fixture-renderer", TokenizerID: "fixture-tokenizer", TokenIDs: ids, PromptTokens: len(ids), ContextWindowTokens: 32768, ReservedOutputTokens: req.OutputTokens, RenderedSHA256: hex.EncodeToString(sum[:])}, nil
}

func (e *normalCorpusEncoder) EstimatedFromCharacters() bool { return false }

func normalCorpusRepo(t *testing.T, enough bool) string {
	t.Helper()
	root := t.TempDir()
	count := 2
	if enough {
		count = 12
	}
	for i := 0; i < count; i++ {
		name := filepath.Join(root, fmt.Sprintf("area%d.go", i))
		body := "package corpus\n\n" + strings.Repeat(fmt.Sprintf("// committed source area %d\n", i), 80)
		if err := os.WriteFile(name, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=Corpus Test", "GIT_AUTHOR_EMAIL=corpus@example.invalid", "GIT_COMMITTER_NAME=Corpus Test", "GIT_COMMITTER_EMAIL=corpus@example.invalid")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	run("init", "-q")
	run("add", ".")
	run("commit", "-q", "-m", "fixture")
	return root
}
