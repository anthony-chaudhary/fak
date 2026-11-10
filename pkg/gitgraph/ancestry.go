package gitgraph

import (
	"bytes"
	"container/heap"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// CommonDir resolves the shared git dir for a path inside a checkout: the
// nearest ancestor holding ".git"; a .git file ("gitdir: <path>", a linked
// worktree or submodule) is followed, then <gitdir>/commondir.
func CommonDir(dir string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	for d := filepath.Clean(abs); ; {
		dotgit := filepath.Join(d, ".git")
		if fi, err := os.Stat(dotgit); err == nil {
			gd := dotgit
			if !fi.IsDir() {
				if gd, err = readGitdirFile(d, dotgit); err != nil {
					return "", err
				}
			}
			return commonDirOf(gd), nil
		}
		parent := filepath.Dir(d)
		if parent == d {
			return "", fmt.Errorf("gitgraph: no .git at or above %s", abs)
		}
		d = parent
	}
}

func readGitdirFile(top, dotgit string) (string, error) {
	data, err := os.ReadFile(dotgit)
	if err != nil {
		return "", err
	}
	line, _, _ := strings.Cut(string(data), "\n")
	rest, ok := strings.CutPrefix(strings.TrimSpace(line), "gitdir:")
	gd := strings.TrimSpace(rest)
	if !ok || gd == "" {
		return "", fmt.Errorf("gitgraph: %s: not a gitdir file", dotgit)
	}
	gd = filepath.FromSlash(gd)
	if !filepath.IsAbs(gd) {
		gd = filepath.Join(top, gd)
	}
	return filepath.Clean(gd), nil
}

func commonDirOf(gitDir string) string {
	data, err := os.ReadFile(filepath.Join(gitDir, "commondir"))
	if err != nil {
		return gitDir
	}
	line, _, _ := strings.Cut(string(data), "\n")
	cd := filepath.FromSlash(strings.TrimSpace(line))
	if cd == "" {
		return gitDir
	}
	if !filepath.IsAbs(cd) {
		cd = filepath.Join(gitDir, cd)
	}
	return filepath.Clean(cd)
}

// plainHistory reports whether commonDir's ancestry is the objects' own. A
// shallow cut, grafts or replace refs change what git answers, so the graph
// (which records raw parents) must not.
func plainHistory(commonDir string) bool {
	if os.Getenv("GIT_GRAFT_FILE") != "" || os.Getenv("GIT_REPLACE_REF_BASE") != "" {
		return false
	}
	for _, p := range []string{"shallow", filepath.Join("info", "grafts"), filepath.Join("refs", "replace")} {
		if _, err := os.Stat(filepath.Join(commonDir, p)); err == nil {
			return false
		}
	}
	packed, _ := os.ReadFile(filepath.Join(commonDir, "packed-refs"))
	return !bytes.Contains(packed, []byte(" refs/replace/"))
}

func isHexOID(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// open loads the graph for dir's repository and resolves full hex commit ids
// to positions; ok is false whenever the graph cannot answer exactly.
func open(dir string, oids ...string) (*Graph, []uint32, bool) {
	for _, o := range oids {
		if !isHexOID(strings.ToLower(o)) {
			return nil, nil, false
		}
	}
	cd, err := CommonDir(dir)
	if err != nil || !plainHistory(cd) {
		return nil, nil, false
	}
	g, err := Load(filepath.Join(cd, "objects"))
	if err != nil {
		return nil, nil, false
	}
	pos := make([]uint32, len(oids))
	for i, o := range oids {
		p, ok := g.Lookup(strings.ToLower(o))
		if !ok {
			return nil, nil, false
		}
		pos[i] = p
	}
	return g, pos, true
}

// IsAncestorInProcess answers `git merge-base --is-ancestor ancestor
// descendant` from the commit-graph. ok is false when the graph is absent,
// unparsable, lacks either commit, lacks generation numbers, or history is
// rewritten (shallow/grafts/replace); the caller must then ask git.
func IsAncestorInProcess(dir, ancestor, descendant string) (is, ok bool) {
	g, pos, ok := open(dir, ancestor, descendant)
	if !ok {
		return false, false
	}
	return g.reaches(pos[1], pos[0])
}

// reaches reports whether target is reachable from start, pruning every
// commit whose generation is below target's: a commit's ancestors all have a
// strictly lower generation, so none of them can be target.
func (g *Graph) reaches(start, target uint32) (bool, bool) {
	if start == target {
		return true, true
	}
	tgen := g.Generation(target)
	if tgen == 0 || g.Generation(start) == 0 {
		return false, false
	}
	if g.Generation(start) <= tgen {
		return false, true
	}
	seen := map[uint32]bool{start: true}
	stack := []uint32{start}
	var parents []uint32
	for len(stack) > 0 {
		c := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		var err error
		parents, err = g.Parents(parents[:0], c)
		if err != nil {
			return false, false
		}
		for _, p := range parents {
			if p == target {
				return true, true
			}
			if seen[p] {
				continue
			}
			seen[p] = true
			pg := g.Generation(p)
			if pg == 0 {
				return false, false
			}
			if pg > tgen {
				stack = append(stack, p)
			}
		}
	}
	return false, true
}

// AheadBehindInProcess counts commits reachable from b but not a (ahead) and
// from a but not b (behind), like `git rev-list --left-right --count a...b`
// reports "behind\tahead". ok follows IsAncestorInProcess.
func AheadBehindInProcess(dir, a, b string) (ahead, behind int, ok bool) {
	g, pos, ok := open(dir, a, b)
	if !ok {
		return 0, 0, false
	}
	return g.aheadBehind(pos[0], pos[1])
}

const (
	flagA uint8 = 1
	flagB uint8 = 2
)

type genHeap struct {
	g     *Graph
	items []uint32
}

func (h *genHeap) Len() int { return len(h.items) }
func (h *genHeap) Less(i, j int) bool {
	gi, gj := h.g.Generation(h.items[i]), h.g.Generation(h.items[j])
	if gi != gj {
		return gi > gj
	}
	return h.items[i] < h.items[j]
}
func (h *genHeap) Swap(i, j int) { h.items[i], h.items[j] = h.items[j], h.items[i] }
func (h *genHeap) Push(x any)    { h.items = append(h.items, x.(uint32)) }
func (h *genHeap) Pop() any {
	n := len(h.items)
	x := h.items[n-1]
	h.items = h.items[:n-1]
	return x
}

// aheadBehind paints both sides down in decreasing generation order, so every
// commit's flags are final when it is popped; the walk stops once every queued
// commit is reachable from both sides.
func (g *Graph) aheadBehind(a, b uint32) (int, int, bool) {
	if g.Generation(a) == 0 || g.Generation(b) == 0 {
		return 0, 0, false
	}
	flags := map[uint32]uint8{}
	flags[a] |= flagA
	flags[b] |= flagB
	h := &genHeap{g: g}
	heap.Push(h, a)
	if b != a {
		heap.Push(h, b)
	}
	pending := 0 // queued commits not yet reachable from both sides
	for _, c := range h.items {
		if flags[c] != flagA|flagB {
			pending++
		}
	}
	ahead, behind := 0, 0
	var parents []uint32
	for h.Len() > 0 && pending > 0 {
		c := heap.Pop(h).(uint32)
		f := flags[c]
		switch f {
		case flagA:
			behind++
			pending--
		case flagB:
			ahead++
			pending--
		}
		var err error
		parents, err = g.Parents(parents[:0], c)
		if err != nil {
			return 0, 0, false
		}
		for _, p := range parents {
			if g.Generation(p) == 0 {
				return 0, 0, false
			}
			old, queued := flags[p]
			nf := old | f
			if queued && nf == old {
				continue
			}
			flags[p] = nf
			switch {
			case !queued:
				heap.Push(h, p)
				if nf != flagA|flagB {
					pending++
				}
			case nf == flagA|flagB:
				pending-- // was queued single-sided, now stale
			}
		}
	}
	return ahead, behind, true
}

// IsAncestor answers in-process when the graph can, else runs
// `git merge-base --is-ancestor` in dir.
func IsAncestor(ctx context.Context, dir, ancestor, descendant string) bool {
	if is, ok := IsAncestorInProcess(dir, ancestor, descendant); ok {
		return is
	}
	return isAncestorGit(ctx, dir, ancestor, descendant)
}

func isAncestorGit(ctx context.Context, dir, ancestor, descendant string) bool {
	cmd := exec.CommandContext(ctx, "git", "merge-base", "--is-ancestor", ancestor, descendant)
	cmd.Dir = dir
	return cmd.Run() == nil
}

// AheadBehind answers in-process when the graph can, else runs
// `git rev-list --left-right --count a...b` in dir.
func AheadBehind(ctx context.Context, dir, a, b string) (ahead, behind int, err error) {
	if ah, bh, ok := AheadBehindInProcess(dir, a, b); ok {
		return ah, bh, nil
	}
	cmd := exec.CommandContext(ctx, "git", "rev-list", "--left-right", "--count", a+"..."+b)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, err
	}
	f := strings.Fields(string(out))
	if len(f) != 2 {
		return 0, 0, errors.New("gitgraph: unexpected rev-list output")
	}
	if behind, err = strconv.Atoi(f[0]); err != nil {
		return 0, 0, err
	}
	if ahead, err = strconv.Atoi(f[1]); err != nil {
		return 0, 0, err
	}
	return ahead, behind, nil
}
