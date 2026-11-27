package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fastImportRepo(t *testing.T, dir, stream string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := newGitCmd(ctx, dir, "fast-import", "--quiet", "--done")
	cmd.Stdin = strings.NewReader(stream)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture fast-import: %v: %s", err, out)
	}
}

// fak-test:runtime medium est=5s lane=default
func TestPeerWIPMassDeletionPartialCheckoutIgnored(t *testing.T) {
	t.Run("pure-mass-deletion-ignored-with-diagnostic", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)

		// Build a base commit with 120 files under docs/ and a corrupt partial-checkout
		// checkpoint that deletes all 120 files with 0 additions/modifications.
		var stream strings.Builder
		stream.WriteString("commit refs/heads/main\nmark :1\ncommitter Fixture <fixture@example.invalid> 1700000000 +0000\ndata 4\nbase\n")
		for i := 0; i < 120; i++ {
			body := fmt.Sprintf("doc %d\n", i)
			fmt.Fprintf(&stream, "M 100644 inline docs/f%03d.txt\ndata %d\n%s", i, len(body), body)
		}
		stream.WriteString("commit refs/fak/wip/ticket-partial-checkout\nmark :2\ncommitter Fixture <fixture@example.invalid> 1700000001 +0000\ndata 26\npreserve partial checkout\nfrom :1\n")
		for i := 0; i < 120; i++ {
			fmt.Fprintf(&stream, "D docs/f%03d.txt\n", i)
		}
		stream.WriteString("done\n")
		fastImportRepo(t, f.dir, stream.String())
		f.git("reset", "--hard", "refs/heads/main")
		corruptOID := f.git("rev-parse", "refs/fak/wip/ticket-partial-checkout")

		// Modify one of the deleted paths in the working tree.
		target := "docs/f005.txt"
		if err := os.WriteFile(filepath.Join(f.dir, filepath.FromSlash(target)), []byte("my edit\n"), 0o600); err != nil {
			t.Fatal(err)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		res, err := ValidatePathAttribution(ctx, realRunner, f.dir, []string{target}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("ValidatePathAttribution: %v", err)
		}
		if !res.OK || res.Reason != "" || len(res.CollidingPaths) != 0 {
			t.Fatalf("mass-deletion partial-checkout ref must not cause PEER_WIP_COLLISION: %+v", res)
		}
		if len(res.MalformedPeerRefs) != 1 {
			t.Fatalf("expected 1 MalformedPeerRefs diagnostic, got %v", res.MalformedPeerRefs)
		}
		diag := res.MalformedPeerRefs[0]
		if !strings.Contains(diag, "refs/fak/wip/ticket-partial-checkout") || !strings.Contains(diag, corruptOID) {
			t.Fatalf("diagnostic must name ref and OID, got %q", diag)
		}
	})

	t.Run("mixed-partial-checkout-protects-real-additions", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)

		// 120 deletions in docs/ plus 2 real additions in internal/compute/.
		var stream strings.Builder
		stream.WriteString("commit refs/heads/main\nmark :1\ncommitter Fixture <fixture@example.invalid> 1700000000 +0000\ndata 4\nbase\n")
		for i := 0; i < 120; i++ {
			body := fmt.Sprintf("doc %d\n", i)
			fmt.Fprintf(&stream, "M 100644 inline docs/f%03d.txt\ndata %d\n%s", i, len(body), body)
		}
		stream.WriteString("commit refs/fak/wip/ticket-sparse-wip\nmark :2\ncommitter Fixture <fixture@example.invalid> 1700000001 +0000\ndata 18\ncheckpoint sparse\nfrom :1\n")
		for i := 0; i < 120; i++ {
			fmt.Fprintf(&stream, "D docs/f%03d.txt\n", i)
		}
		addBody := "package compute\n"
		fmt.Fprintf(&stream, "M 100644 inline internal/compute/new.go\ndata %d\n%s", len(addBody), addBody)
		fmt.Fprintf(&stream, "M 100644 inline internal/compute/new_test.go\ndata %d\n%s", len(addBody), addBody)
		stream.WriteString("done\n")
		fastImportRepo(t, f.dir, stream.String())
		f.git("reset", "--hard", "refs/heads/main")
		corruptOID := f.git("rev-parse", "refs/fak/wip/ticket-sparse-wip")

		// 1. Editing a deleted docs/ path is admitted with a diagnostic warning.
		if err := os.WriteFile(filepath.Join(f.dir, filepath.FromSlash("docs/f010.txt")), []byte("my doc edit\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		res, err := ValidatePathAttribution(ctx, realRunner, f.dir, []string{"docs/f010.txt"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("ValidatePathAttribution(docs): %v", err)
		}
		if !res.OK || len(res.MalformedPeerRefs) != 1 || !strings.Contains(res.MalformedPeerRefs[0], corruptOID) {
			t.Fatalf("deleted path in sparse checkout should be ignored with diagnostic: %+v", res)
		}

		// 2. Touching the genuinely added file still triggers PEER_WIP_COLLISION.
		addedPath := filepath.Join(f.dir, filepath.FromSlash("internal/compute/new.go"))
		if err := os.MkdirAll(filepath.Dir(addedPath), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(addedPath, []byte("package compute // conflict\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		resAdd, err := ValidatePathAttribution(ctx, realRunner, f.dir, []string{"internal/compute/new.go"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("ValidatePathAttribution(added): %v", err)
		}
		if resAdd.OK || resAdd.Reason != ReasonPeerWIPCollision {
			t.Fatalf("genuinely added path in sparse checkpoint must still collide: %+v", resAdd)
		}
	})

	t.Run("over-500-deletions-respects-target-dir-non-D-changes", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)

		// 510 deletions in docs/, 1 deletion in internal/active/, and 60 modifications in internal/active/.
		// Here deletions=511, nonDeletions=60 (511 <= 10*60), so only the >500 + 0 non-D-in-dir rule applies.
		var stream strings.Builder
		stream.WriteString("commit refs/heads/main\nmark :1\ncommitter Fixture <fixture@example.invalid> 1700000000 +0000\ndata 4\nbase\n")
		for i := 0; i < 510; i++ {
			body := fmt.Sprintf("d%d\n", i)
			fmt.Fprintf(&stream, "M 100644 inline docs/f%03d.txt\ndata %d\n%s", i, len(body), body)
		}
		for i := 0; i < 61; i++ {
			body := fmt.Sprintf("a%d\n", i)
			fmt.Fprintf(&stream, "M 100644 inline internal/active/a%02d.go\ndata %d\n%s", i, len(body), body)
		}
		stream.WriteString("commit refs/fak/wip/ticket-big-sparse\nmark :2\ncommitter Fixture <fixture@example.invalid> 1700000001 +0000\ndata 10\nbig sparse\nfrom :1\n")
		for i := 0; i < 510; i++ {
			fmt.Fprintf(&stream, "D docs/f%03d.txt\n", i)
		}
		stream.WriteString("D internal/active/a60.go\n")
		for i := 0; i < 60; i++ {
			body := fmt.Sprintf("mod %d\n", i)
			fmt.Fprintf(&stream, "M 100644 inline internal/active/a%02d.go\ndata %d\n%s", i, len(body), body)
		}
		stream.WriteString("done\n")
		fastImportRepo(t, f.dir, stream.String())
		f.git("reset", "--hard", "refs/heads/main")

		// docs/ has 0 non-D changes -> ignored.
		if err := os.WriteFile(filepath.Join(f.dir, filepath.FromSlash("docs/f001.txt")), []byte("edit\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		resDoc, err := ValidatePathAttribution(ctx, realRunner, f.dir, []string{"docs/f001.txt"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("ValidatePathAttribution(docs): %v", err)
		}
		if !resDoc.OK || len(resDoc.MalformedPeerRefs) != 1 {
			t.Fatalf("docs deletion with 0 non-D in docs should be ignored: %+v", resDoc)
		}

		// internal/active/ has 60 non-D changes -> deletion in internal/active/ still collides.
		if err := os.WriteFile(filepath.Join(f.dir, filepath.FromSlash("internal/active/a60.go")), []byte("edit\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		resAct, err := ValidatePathAttribution(ctx, realRunner, f.dir, []string{"internal/active/a60.go"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("ValidatePathAttribution(active): %v", err)
		}
		if resAct.OK || resAct.Reason != ReasonPeerWIPCollision {
			t.Fatalf("deletion in active peer dir must still collide: %+v", resAct)
		}
	})

	t.Run("small-legitimate-deletion-still-collides", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)

		var stream strings.Builder
		stream.WriteString("commit refs/heads/main\nmark :1\ncommitter Fixture <fixture@example.invalid> 1700000000 +0000\ndata 4\nbase\n")
		for i := 0; i < 3; i++ {
			body := fmt.Sprintf("f%d\n", i)
			fmt.Fprintf(&stream, "M 100644 inline pkg/f%d.go\ndata %d\n%s", i, len(body), body)
		}
		stream.WriteString("commit refs/fak/wip/ticket-small-del\nmark :2\ncommitter Fixture <fixture@example.invalid> 1700000001 +0000\ndata 10\nremove old\nfrom :1\n")
		for i := 0; i < 3; i++ {
			fmt.Fprintf(&stream, "D pkg/f%d.go\n", i)
		}
		stream.WriteString("done\n")
		fastImportRepo(t, f.dir, stream.String())
		f.git("reset", "--hard", "refs/heads/main")

		if err := os.WriteFile(filepath.Join(f.dir, filepath.FromSlash("pkg/f1.go")), []byte("edit\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		res, err := ValidatePathAttribution(ctx, realRunner, f.dir, []string{"pkg/f1.go"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("ValidatePathAttribution: %v", err)
		}
		if res.OK || res.Reason != ReasonPeerWIPCollision || len(res.MalformedPeerRefs) != 0 {
			t.Fatalf("small legitimate deletion must still collide without malformed warning: %+v", res)
		}
	})
}

// fak-test:runtime medium est=5s lane=default
func TestPeerWIPUnstampedTicketRepresented(t *testing.T) {
	paths := []string{"owned/a.txt", "owned/b.txt"}
	baseEntries := peerHistoryEntries("100644", "base-a\n", "base-b\n")
	payloadEntries := peerHistoryEntries("100644", "payload-a\n", "payload-b\n")
	headEntries := peerHistoryEntries("100644", "head-a\n", "head-b\n")

	t.Run("unstamped-ticket-landed-at-head-squashed", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		// Checkpoint sits on an intermediate ticket commit not on main, mimicking a squashed landing.
		branchBase := f.commit([]string{base}, "ticket intermediate", peerHistoryEntries("100644", "mid-a\n", "base-b\n"))
		checkpoint := f.commit([]string{branchBase}, "checkpoint: preserve supervised worker WIP", payloadEntries)
		f.git("update-ref", "refs/fak/wip/ticket-landed-head", checkpoint)
		// main squashed the same final blobs onto base.
		head := f.commit([]string{base}, "feat: squash landed ticket", payloadEntries)
		f.git("update-ref", "HEAD", head)

		owners, err := f.resolve(t, realRunner, paths)
		if err != nil {
			t.Fatal(err)
		}
		if len(owners) != 0 {
			t.Fatalf("unstamped ticket-* checkpoint already at HEAD retained ownership: %+v", owners)
		}
	})

	t.Run("unstamped-ticket-landed-at-earlier-ancestor", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, "checkpoint: preserve supervised worker WIP", payloadEntries)
		f.git("update-ref", "refs/fak/wip/ticket-landed-ancestor", checkpoint)
		landed := f.commit([]string{base}, "feat: land ticket", payloadEntries)
		head := f.commit([]string{landed}, "chore: later edit", headEntries)
		f.git("update-ref", "HEAD", head)

		owners, err := f.resolve(t, realRunner, paths)
		if err != nil {
			t.Fatal(err)
		}
		if len(owners) != 0 {
			t.Fatalf("unstamped ticket-* checkpoint landed at ancestor retained ownership: %+v", owners)
		}
	})

	t.Run("unstamped-ticket-unlanded-retains-ownership", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, "checkpoint: preserve supervised worker WIP", payloadEntries)
		f.git("update-ref", "refs/fak/wip/ticket-unlanded", checkpoint)
		head := f.commit([]string{base}, "unrelated advance", headEntries)
		f.git("update-ref", "HEAD", head)

		owners, err := f.resolve(t, realRunner, paths)
		if err != nil {
			t.Fatal(err)
		}
		if owners["owned/a.txt"].Peer != "ticket-unlanded" || owners["owned/b.txt"].Peer != "ticket-unlanded" {
			t.Fatalf("unlanded unstamped ticket-* checkpoint must retain ownership: %+v", owners)
		}
	})

	t.Run("unstamped-ticket-partial-landing-retains-ownership", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, "checkpoint: preserve supervised worker WIP", payloadEntries)
		f.git("update-ref", "refs/fak/wip/ticket-partial", checkpoint)
		partial := f.commit([]string{base}, "only a landed", peerHistoryEntries("100644", "payload-a\n", "base-b\n"))
		f.git("update-ref", "HEAD", partial)

		owners, err := f.resolve(t, realRunner, paths)
		if err != nil {
			t.Fatal(err)
		}
		if owners["owned/a.txt"].Peer != "ticket-partial" || owners["owned/b.txt"].Peer != "ticket-partial" {
			t.Fatalf("partially landed unstamped ticket-* checkpoint must retain ownership: %+v", owners)
		}
	})

	t.Run("malformed-stamp-ticket-fails-closed", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, "fak-wip: {malformed", payloadEntries)
		f.git("update-ref", "refs/fak/wip/ticket-badstamp", checkpoint)
		head := f.commit([]string{base}, "land payload", payloadEntries)
		f.git("update-ref", "HEAD", head)

		owners, err := f.resolve(t, realRunner, paths)
		if err != nil {
			t.Fatal(err)
		}
		if owners["owned/a.txt"].Peer != "ticket-badstamp" || owners["owned/b.txt"].Peer != "ticket-badstamp" {
			t.Fatalf("malformed-stamp ticket-* checkpoint must fail closed: %+v", owners)
		}
	})
}
