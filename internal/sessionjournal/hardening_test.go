package sessionjournal

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"testing"
)

func TestConcurrentAppendSerializesCompleteRows(t *testing.T) {
	path := t.TempDir() + "/journal.jsonl"
	const writers = 32
	var wg sync.WaitGroup
	errs := make(chan error, writers)
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs <- Append(path, Event{Schema: Schema, Kind: KindOpen, ID: fmt.Sprintf("s-%02d", i), CWD: t.TempDir()})
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	events := LoadFile(path)
	if len(events) != writers {
		t.Fatalf("events=%d want %d", len(events), writers)
	}
	seen := map[string]bool{}
	for _, event := range events {
		seen[event.ID] = true
	}
	if len(seen) != writers {
		t.Fatalf("unique sessions=%d want %d", len(seen), writers)
	}
}

func TestCrossProcessAppendSerializesCompleteRows(t *testing.T) {
	path := t.TempDir() + "/journal.jsonl"
	const writers = 12
	commands := make([]*exec.Cmd, writers)
	for i := range commands {
		commands[i] = exec.Command(os.Args[0], "-test.run=^TestAppendHelperProcess$", "--", path, fmt.Sprintf("p-%02d", i))
		commands[i].Env = append(os.Environ(), "FAK_SESSION_JOURNAL_APPEND_HELPER=1")
		if err := commands[i].Start(); err != nil {
			t.Fatal(err)
		}
	}
	for _, command := range commands {
		if err := command.Wait(); err != nil {
			t.Fatalf("append helper: %v", err)
		}
	}
	events := LoadFile(path)
	if len(events) != writers {
		t.Fatalf("cross-process events=%d want %d", len(events), writers)
	}
}

func TestAppendHelperProcess(t *testing.T) {
	if os.Getenv("FAK_SESSION_JOURNAL_APPEND_HELPER") != "1" {
		return
	}
	separator := 0
	for i, arg := range os.Args {
		if arg == "--" {
			separator = i
			break
		}
	}
	if separator == 0 || len(os.Args) != separator+3 {
		t.Fatalf("helper arguments: %q", os.Args)
	}
	if err := Append(os.Args[separator+1], Event{Schema: Schema, Kind: KindOpen, ID: os.Args[separator+2], CWD: os.TempDir()}); err != nil {
		t.Fatal(err)
	}
}

func TestCompactDropsTornRowsAndRemainsAppendable(t *testing.T) {
	path := t.TempDir() + "/journal.jsonl"
	if err := Append(path, Event{Schema: Schema, Kind: KindOpen, ID: "before", CWD: t.TempDir()}); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.WriteString("{torn\n")
	_ = f.Close()
	if n, err := Compact(path); err != nil || n != 1 {
		t.Fatalf("Compact=(%d,%v), want (1,nil)", n, err)
	}
	if err := Append(path, Event{Schema: Schema, Kind: KindClose, ID: "before"}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "torn") || len(ParseEvents(string(b))) != 2 {
		t.Fatalf("unexpected compacted journal: %s", b)
	}
}
