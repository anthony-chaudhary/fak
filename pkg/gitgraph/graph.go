// Package gitgraph answers commit ancestry queries in-process from git's
// commit-graph files (https://git-scm.com/docs/commit-graph-format), so a hot
// loop of `git merge-base --is-ancestor` checks costs a map lookup and a
// generation-pruned walk instead of one process spawn each (0.3-1.3 s on a
// loaded Windows host).
//
// Every in-process answer is exact or absent: a missing, stale (commit not in
// the graph), unparsable, or history-rewriting setup (shallow, grafts, replace
// refs) yields ok=false and the caller asks git.
package gitgraph

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	graphSignature = "CGPH"
	noParent       = 0x70000000
	edgeFlag       = 0x80000000
	edgeLast       = 0x80000000
	chunkFanout    = 0x4f494446 // OIDF
	chunkOIDs      = 0x4f49444c // OIDL
	chunkData      = 0x43444154 // CDAT
	chunkEdges     = 0x45444745 // EDGE
)

// layer is one commit-graph file. Positions are global: the layer's first
// commit sits at base, the sum of the commit counts of the layers below it.
type layer struct {
	hashLen int
	fanout  []byte
	oids    []byte
	data    []byte
	edges   []byte
	count   uint32
	base    uint32
}

// Graph is a parsed commit-graph, a single file or a split chain.
type Graph struct {
	layers  []*layer // base first
	hashLen int
	total   uint32
}

// errNoGraph marks a repository without any commit-graph file.
var errNoGraph = errors.New("gitgraph: no commit-graph")

func parseLayer(buf []byte, base uint32) (*layer, int, error) {
	if len(buf) < 8 || string(buf[:4]) != graphSignature {
		return nil, 0, errors.New("gitgraph: bad signature")
	}
	if buf[4] != 1 {
		return nil, 0, fmt.Errorf("gitgraph: unsupported version %d", buf[4])
	}
	var hashLen int
	switch buf[5] {
	case 1:
		hashLen = 20
	case 2:
		hashLen = 32
	default:
		return nil, 0, fmt.Errorf("gitgraph: unknown hash version %d", buf[5])
	}
	nChunks := int(buf[6])
	nBase := int(buf[7])
	toc := 8
	if len(buf) < toc+(nChunks+1)*12 {
		return nil, 0, errors.New("gitgraph: truncated chunk table")
	}
	l := &layer{hashLen: hashLen, base: base}
	for i := 0; i < nChunks; i++ {
		e := buf[toc+i*12:]
		id := binary.BigEndian.Uint32(e[0:4])
		start := binary.BigEndian.Uint64(e[4:12])
		end := binary.BigEndian.Uint64(e[16:24])
		if start > end || end > uint64(len(buf)) {
			return nil, 0, errors.New("gitgraph: chunk out of range")
		}
		c := buf[start:end]
		switch id {
		case chunkFanout:
			l.fanout = c
		case chunkOIDs:
			l.oids = c
		case chunkData:
			l.data = c
		case chunkEdges:
			l.edges = c
		}
	}
	if len(l.fanout) != 256*4 || l.oids == nil || l.data == nil {
		return nil, 0, errors.New("gitgraph: missing required chunk")
	}
	l.count = binary.BigEndian.Uint32(l.fanout[255*4:])
	if uint64(len(l.oids)) != uint64(l.count)*uint64(hashLen) ||
		uint64(len(l.data)) != uint64(l.count)*uint64(hashLen+16) {
		return nil, 0, errors.New("gitgraph: chunk size mismatch")
	}
	return l, nBase, nil
}

// find returns the local position of oid in the layer.
func (l *layer) find(oid []byte) (uint32, bool) {
	lo := uint32(0)
	if oid[0] > 0 {
		lo = binary.BigEndian.Uint32(l.fanout[(int(oid[0])-1)*4:])
	}
	hi := binary.BigEndian.Uint32(l.fanout[int(oid[0])*4:])
	if hi > l.count || lo > hi {
		return 0, false
	}
	for lo < hi {
		mid := lo + (hi-lo)/2
		c := bytes.Compare(l.oids[int(mid)*l.hashLen:int(mid+1)*l.hashLen], oid)
		switch {
		case c == 0:
			return mid, true
		case c < 0:
			lo = mid + 1
		default:
			hi = mid
		}
	}
	return 0, false
}

// Lookup returns the global graph position of a hex commit id.
func (g *Graph) Lookup(hexOID string) (uint32, bool) {
	if len(hexOID) != 2*g.hashLen {
		return 0, false
	}
	oid, err := hex.DecodeString(hexOID)
	if err != nil {
		return 0, false
	}
	for i := len(g.layers) - 1; i >= 0; i-- {
		if p, ok := g.layers[i].find(oid); ok {
			return g.layers[i].base + p, true
		}
	}
	return 0, false
}

func (g *Graph) layerOf(pos uint32) *layer {
	for i := len(g.layers) - 1; i >= 0; i-- {
		if pos >= g.layers[i].base {
			return g.layers[i]
		}
	}
	return nil
}

// Generation is the topological level (generation number v1) of pos; zero
// means the writer did not compute it.
func (g *Graph) Generation(pos uint32) uint32 {
	l := g.layerOf(pos)
	rec := l.data[int(pos-l.base)*(l.hashLen+16)+l.hashLen:]
	return binary.BigEndian.Uint32(rec[8:12]) >> 2
}

// Parents appends the global positions of pos's parents to dst.
func (g *Graph) Parents(dst []uint32, pos uint32) ([]uint32, error) {
	l := g.layerOf(pos)
	rec := l.data[int(pos-l.base)*(l.hashLen+16)+l.hashLen:]
	p1 := binary.BigEndian.Uint32(rec[0:4])
	p2 := binary.BigEndian.Uint32(rec[4:8])
	if p1 == noParent {
		return dst, nil
	}
	if p1 >= g.total {
		return dst, errors.New("gitgraph: parent out of range")
	}
	dst = append(dst, p1)
	if p2 == noParent {
		return dst, nil
	}
	if p2&edgeFlag == 0 {
		if p2 >= g.total {
			return dst, errors.New("gitgraph: parent out of range")
		}
		return append(dst, p2), nil
	}
	for i := int(p2 &^ edgeFlag); ; i++ {
		if (i+1)*4 > len(l.edges) {
			return dst, errors.New("gitgraph: edge list out of range")
		}
		e := binary.BigEndian.Uint32(l.edges[i*4:])
		if e&^edgeLast >= g.total {
			return dst, errors.New("gitgraph: parent out of range")
		}
		dst = append(dst, e&^edgeLast)
		if e&edgeLast != 0 {
			return dst, nil
		}
	}
}

// fileSig identifies one on-disk file version for cache revalidation.
type fileSig struct {
	path string
	size int64
	mod  time.Time
}

func statSig(path string) (fileSig, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		return fileSig{}, false
	}
	return fileSig{path: path, size: fi.Size(), mod: fi.ModTime()}, true
}

type cacheEntry struct {
	sigs  []fileSig
	graph *Graph
}

var (
	cacheMu sync.Mutex
	cache   = map[string]cacheEntry{}
)

// graphFiles lists the commit-graph files git would load for objectsDir, base
// first: a chain when objects/info/commit-graphs/commit-graph-chain exists,
// else the single objects/info/commit-graph.
func graphFiles(objectsDir string) ([]string, []fileSig, error) {
	chainDir := filepath.Join(objectsDir, "info", "commit-graphs")
	chainPath := filepath.Join(chainDir, "commit-graph-chain")
	if sig, ok := statSig(chainPath); ok {
		data, err := os.ReadFile(chainPath)
		if err != nil {
			return nil, nil, err
		}
		sigs := []fileSig{sig}
		var files []string
		sc := bufio.NewScanner(bytes.NewReader(data))
		for sc.Scan() {
			h := strings.TrimSpace(sc.Text())
			if h == "" {
				continue
			}
			p := filepath.Join(chainDir, "graph-"+h+".graph")
			s, ok := statSig(p)
			if !ok {
				return nil, nil, fmt.Errorf("gitgraph: chain layer %s missing", h)
			}
			files = append(files, p)
			sigs = append(sigs, s)
		}
		if len(files) == 0 {
			return nil, nil, errors.New("gitgraph: empty chain")
		}
		return files, sigs, nil
	}
	single := filepath.Join(objectsDir, "info", "commit-graph")
	if s, ok := statSig(single); ok {
		return []string{single}, []fileSig{s}, nil
	}
	return nil, nil, errNoGraph
}

func sameSigs(a, b []fileSig) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].path != b[i].path || a[i].size != b[i].size || !a[i].mod.Equal(b[i].mod) {
			return false
		}
	}
	return true
}

// Load returns the commit-graph for objectsDir, reusing a parsed copy while
// every file's size and mtime are unchanged.
func Load(objectsDir string) (*Graph, error) {
	files, sigs, err := graphFiles(objectsDir)
	if err != nil {
		return nil, err
	}
	key := filepath.Clean(objectsDir)
	cacheMu.Lock()
	ent, hit := cache[key]
	cacheMu.Unlock()
	if hit && sameSigs(ent.sigs, sigs) {
		return ent.graph, nil
	}
	g := &Graph{}
	for i, f := range files {
		buf, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		l, nBase, err := parseLayer(buf, g.total)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		if nBase != i {
			return nil, fmt.Errorf("gitgraph: %s: base count %d at chain index %d", f, nBase, i)
		}
		if g.hashLen != 0 && g.hashLen != l.hashLen {
			return nil, errors.New("gitgraph: mixed hash lengths")
		}
		g.hashLen = l.hashLen
		g.layers = append(g.layers, l)
		g.total += l.count
	}
	cacheMu.Lock()
	cache[key] = cacheEntry{sigs: sigs, graph: g}
	cacheMu.Unlock()
	return g, nil
}
