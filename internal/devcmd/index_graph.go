package devcmd

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/docreach"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func indexGraphMain(stdout, stderr io.Writer, root string, args []string) int {
	jsonOut := false
	for _, a := range args {
		if a == "--json" {
			jsonOut = true
		} else {
			fmt.Fprintf(stderr, "fak-dev index graph: unknown argument %s\n", a)
			return 2
		}
	}
	head, err := windowgate.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index graph: resolve HEAD: %v\n", err)
		return 1
	}
	commit := strings.TrimSpace(string(head))
	blobs, err := readCommittedMarkdown(root, commit)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index graph: read HEAD: %v\n", err)
		return 1
	}
	r := docreach.Census(commit, blobs)
	if jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(r); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "commit=%s documents=%d broken_links=%d\n", r.Commit, r.Documents, len(r.BrokenLinks))
	for _, c := range r.Rules {
		fmt.Fprintf(stdout, "%s reached=%d denominator=%d unreached=%d\n", c.Rule, c.Numerator, c.Denominator, len(c.Unreached))
		for _, p := range c.Unreached {
			fmt.Fprintf(stdout, "  %s\n", p)
		}
	}
	for _, b := range r.BrokenLinks {
		fmt.Fprintf(stdout, "BROKEN %s -> %s\n", b.Source, b.Target)
	}
	return 0
}

type markdownObject struct {
	path string
	oid  string
}

// readCommittedMarkdown reads the Markdown corpus from one immutable tree with
// two Git processes regardless of corpus size. ls-tree supplies NUL-delimited
// paths and object IDs; cat-file --batch supplies the corresponding blob bytes.
// The working tree and index are never inputs.
func readCommittedMarkdown(root, commit string) ([]docreach.Blob, error) {
	tree, err := windowgate.Command("git", "-C", root, "ls-tree", "-r", "-z", commit).Output()
	if err != nil {
		return nil, fmt.Errorf("list %s: %w", commit, err)
	}
	var objects []markdownObject
	for _, entry := range bytes.Split(tree, []byte{0}) {
		if len(entry) == 0 {
			continue
		}
		tab := bytes.IndexByte(entry, '\t')
		if tab < 0 {
			return nil, fmt.Errorf("malformed ls-tree entry %q", entry)
		}
		meta := strings.Fields(string(entry[:tab]))
		name := string(entry[tab+1:])
		if len(meta) != 3 || meta[1] != "blob" {
			continue
		}
		if strings.HasSuffix(strings.ToLower(name), ".md") {
			objects = append(objects, markdownObject{path: name, oid: meta[2]})
		}
	}

	cmd := windowgate.Command("git", "-C", root, "cat-file", "--batch")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	var commandErr bytes.Buffer
	cmd.Stderr = &commandErr
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, err
	}
	finished := false
	defer func() {
		if !finished {
			_ = stdin.Close()
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	request := bufio.NewWriter(stdin)
	response := bufio.NewReader(stdout)
	blobs := make([]docreach.Blob, 0, len(objects))
	for _, object := range objects {
		if _, err := fmt.Fprintln(request, object.oid); err != nil {
			return nil, err
		}
		if err := request.Flush(); err != nil {
			return nil, err
		}
		header, err := response.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("read %s header: %w", object.path, err)
		}
		fields := strings.Fields(header)
		if len(fields) != 3 || fields[1] != "blob" {
			return nil, fmt.Errorf("read %s: unexpected cat-file header %q", object.path, strings.TrimSpace(header))
		}
		size, err := strconv.ParseInt(fields[2], 10, 64)
		if err != nil || size < 0 {
			return nil, fmt.Errorf("read %s: invalid blob size %q", object.path, fields[2])
		}
		body := make([]byte, size)
		if _, err := io.ReadFull(response, body); err != nil {
			return nil, fmt.Errorf("read %s body: %w", object.path, err)
		}
		delim, err := response.ReadByte()
		if err != nil {
			return nil, fmt.Errorf("read %s delimiter: %w", object.path, err)
		}
		if delim != '\n' {
			return nil, fmt.Errorf("read %s delimiter: got %#x, want newline", object.path, delim)
		}
		blobs = append(blobs, docreach.Blob{Path: object.path, Text: string(body)})
	}
	if err := stdin.Close(); err != nil {
		return nil, err
	}
	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("git cat-file: %w: %s", err, strings.TrimSpace(commandErr.String()))
	}
	finished = true
	return blobs, nil
}

func boolArg(v bool) []string {
	if v {
		return []string{"--json"}
	}
	return nil
}
