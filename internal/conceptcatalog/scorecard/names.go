package scorecard

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// ---------------------------------------------------------------------------
// Names.
// ---------------------------------------------------------------------------

// bareName strips a trailing parenthetical gloss: re.sub(r"\s*\([^)]*\)\s*$", "", s).strip().
func bareName(v Value) string {
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return pyStrip(stripGloss(s))
}

func stripGloss(s string) string {
	end := len(s)
	for end > 0 {
		r, size := utf8.DecodeLastRuneInString(s[:end])
		if !pyIsSpace(r) {
			break
		}
		end -= size
	}
	if end == 0 || s[end-1] != ')' {
		return s
	}
	closing := end - 1
	prev := strings.LastIndexByte(s[:closing], ')')
	open := strings.IndexByte(s[prev+1:closing], '(')
	if open < 0 {
		return s
	}
	open += prev + 1
	start := open
	for start > 0 {
		r, size := utf8.DecodeLastRuneInString(s[:start])
		if !pyIsSpace(r) {
			break
		}
		start -= size
	}
	return s[:start]
}

func isAlnumASCII(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

// splitWords lowercases the word pieces of a name across camelCase, ALLCAPS,
// snake, dotted and spaced spellings, as the Python word regex does.
func splitWords(v Value) []string {
	s := bareName(v)
	var out []string
	i := 0
	for i < len(s) {
		if !isAlnumASCII(s[i]) {
			i++
			continue
		}
		j := i
		for j < len(s) && isAlnumASCII(s[j]) {
			j++
		}
		out = append(out, wordPieces(s[i:j])...)
		i = j
	}
	return out
}

func isUpper(c byte) bool      { return c >= 'A' && c <= 'Z' }
func isLowerDigit(c byte) bool { return c >= 'a' && c <= 'z' || c >= '0' && c <= '9' }

// wordPieces is findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+") over one
// alphanumeric chunk, lowercased.
func wordPieces(chunk string) []string {
	var out []string
	i := 0
	for i < len(chunk) {
		c := chunk[i]
		if isUpper(c) {
			j := i
			for j < len(chunk) && isUpper(chunk[j]) {
				j++
			}
			lowerNext := j < len(chunk) && chunk[j] >= 'a' && chunk[j] <= 'z'
			switch {
			case !lowerNext:
				out = append(out, strings.ToLower(chunk[i:j]))
				i = j
			case j-i >= 2:
				out = append(out, strings.ToLower(chunk[i:j-1]))
				i = j - 1
			default:
				k := i + 1
				for k < len(chunk) && isLowerDigit(chunk[k]) {
					k++
				}
				out = append(out, strings.ToLower(chunk[i:k]))
				i = k
			}
			continue
		}
		j := i
		for j < len(chunk) && isLowerDigit(chunk[j]) {
			j++
		}
		out = append(out, chunk[i:j])
		i = j
	}
	return out
}

func editDistanceWithin(a, b string, limit int) int {
	if abs(len(a)-len(b)) > limit {
		return limit + 1
	}
	prev := make([]int, len(b)+1)
	cur := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(a); i++ {
		cur[0] = i
		best := i
		for j := 1; j <= len(b); j++ {
			sub := prev[j-1]
			if a[i-1] != b[j-1] {
				sub++
			}
			v := min(prev[j]+1, cur[j-1]+1, sub)
			cur[j] = v
			if v < best {
				best = v
			}
		}
		if best > limit {
			return limit + 1
		}
		prev, cur = cur, prev
	}
	return prev[len(b)]
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func sharedAffix(a, b string) int {
	head := 0
	for head < len(a) && head < len(b) && a[head] == b[head] {
		head++
	}
	tail := 0
	for tail < len(a) && tail < len(b) && a[len(a)-1-tail] == b[len(b)-1-tail] {
		tail++
	}
	return max(head, tail)
}

// isWireKey: '.', '-', or ALLCAPS_WITH_UNDERSCORE (a trailing newline is
// tolerated, as `$` allows under re.match).
func isWireKey(v Value) bool {
	s, ok := v.(string)
	if !ok {
		return false
	}
	if strings.ContainsAny(s, ".-") {
		return true
	}
	s = strings.TrimSuffix(s, "\n")
	parts := strings.Split(s, "_")
	if len(parts) < 2 {
		return false
	}
	for _, p := range parts {
		if p == "" {
			return false
		}
		for i := 0; i < len(p); i++ {
			if !(p[i] >= 'A' && p[i] <= 'Z' || p[i] >= '0' && p[i] <= '9') {
				return false
			}
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Collisions, clusters, separation edges, pairs.
// ---------------------------------------------------------------------------

type collision struct {
	id     string
	others []string
}

func findCollisions(rows []*Object) ([]collision, map[string]bool) {
	order := []string{}
	byCanon := map[string][]string{}
	for i, r := range rows {
		rid := ridTruthy(r, i)
		c := normValue(getOr(r, "canonical", ""))
		if c == "" {
			continue
		}
		if _, ok := byCanon[c]; !ok {
			order = append(order, c)
		}
		byCanon[c] = append(byCanon[c], rid)
	}
	out := map[string][]string{}
	keys := []string{}
	for _, c := range order {
		ids := byCanon[c]
		if len(ids) < 2 {
			continue
		}
		for _, rid := range ids {
			var others []string
			for _, o := range ids {
				if o != rid {
					others = append(others, o)
				}
			}
			if _, ok := out[rid]; !ok {
				keys = append(keys, rid)
			}
			out[rid] = nonNil(others)
		}
	}
	set := map[string]bool{}
	list := make([]collision, 0, len(keys))
	for _, k := range keys {
		set[k] = true
		list = append(list, collision{k, out[k]})
	}
	return list, set
}

func clusterSizes(rows []*Object) map[string]int {
	sizes := map[string]int{}
	for _, r := range rows {
		fam := get(r, "family")
		if nonempty(fam) {
			sizes[fam.(string)]++
		}
	}
	return sizes
}

func familySize(sizes map[string]int, fam Value) int {
	if s, ok := fam.(string); ok {
		return sizes[s]
	}
	return 0
}

func idSet(rows []*Object) map[string]bool {
	ids := map[string]bool{}
	for _, r := range rows {
		if id := get(r, "id"); nonempty(id) {
			ids[id.(string)] = true
		}
	}
	return ids
}

func separationEdges(rows []*Object, byKey map[string][]string) (edgeSet, []string) {
	ids := idSet(rows)
	type canonTarget struct {
		id    string
		valid bool
	}
	byCanon := map[string]canonTarget{}
	for _, r := range rows {
		if !nonempty(get(r, "canonical")) {
			continue
		}
		id := get(r, "id")
		t := canonTarget{}
		if id != nil {
			t = canonTarget{pyStr(id), true}
		}
		byCanon[normValue(get(r, "canonical"))] = t
	}
	edges := edgeSet{}
	unresolved := []string{}
	for i, r := range rows {
		rid := ridIndexed(r, i)
		for _, refV := range iterValue(orEmptyList(get(r, "distinct_from"))) {
			if !nonempty(refV) {
				continue
			}
			ref := refV.(string)
			target, found := "", false
			if ids[ref] {
				target, found = ref, true
			} else if t, ok := byCanon[normToken(ref)]; ok && t.valid {
				target, found = t.id, true
			}
			if !found {
				var landing []string
				for _, t := range byKey[normToken(bareName(ref))] {
					if t != rid {
						landing = append(landing, t)
					}
				}
				if len(landing) == 1 {
					target, found = landing[0], true
				} else if len(landing) > 1 {
					shown := landing
					if len(shown) > 4 {
						shown = shown[:4]
					}
					more := ""
					if len(landing) > 4 {
						more = "..."
					}
					unresolved = append(unresolved, fmt.Sprintf(
						"%s: distinct_from %s names %d concepts at once (%s%s) - an ambiguous boundary; use the catalog id of the one you mean",
						rid, pyStrString(ref), len(landing), strings.Join(shown, ", "), more))
					continue
				}
			}
			if !found {
				unresolved = append(unresolved, fmt.Sprintf(
					"%s: distinct_from %s resolves to no catalog id, canonical name or index key - a dangling boundary",
					rid, pyStrString(ref)))
			} else if target != rid {
				edges[[2]string{rid, target}] = true
			}
		}
	}
	return edges, unresolved
}

// Pair is a discovered confusable name-pair.
type Pair struct {
	A, B, Kind, Why string
}

func confusablePairs(rows []*Object) []Pair {
	type prepared struct {
		id    string
		tok   string
		words []string
	}
	var named []prepared
	for _, r := range rows {
		if !nonempty(get(r, "id")) || !nonempty(get(r, "canonical")) {
			continue
		}
		c := get(r, "canonical")
		named = append(named, prepared{get(r, "id").(string), normToken(bareName(c)), splitWords(c)})
	}
	found := map[[2]string]Pair{}
	add := func(a, b, kind, why string) {
		key := [2]string{a, b}
		if a >= b {
			key = [2]string{b, a}
		}
		cur, ok := found[key]
		if !ok || pairRank[kind] < pairRank[cur.Kind] {
			found[key] = Pair{key[0], key[1], kind, why}
		}
	}
	addAll := func(ids []string, kind, why string) {
		sorted := append([]string(nil), ids...)
		sort.Strings(sorted)
		for i, a := range sorted {
			for _, b := range sorted[i+1:] {
				add(a, b, kind, why)
			}
		}
	}

	var bareOrder []string
	byBare := map[string][]string{}
	for _, p := range named {
		if p.tok == "" {
			continue
		}
		if _, ok := byBare[p.tok]; !ok {
			bareOrder = append(bareOrder, p.tok)
		}
		byBare[p.tok] = append(byBare[p.tok], p.id)
	}
	for _, tok := range bareOrder {
		if ids := byBare[tok]; len(ids) >= 2 {
			addAll(ids, "homonym", fmt.Sprintf("both are plainly named %s once the gloss is stripped", pyStrString(tok)))
		}
	}

	var wordOrder []string
	byWords := map[string][]string{}
	wordText := map[string]string{}
	for _, p := range named {
		if len(p.words) < 2 {
			continue
		}
		ws := append([]string(nil), p.words...)
		sort.Strings(ws)
		key := strings.Join(ws, "\x00")
		if _, ok := byWords[key]; !ok {
			wordOrder = append(wordOrder, key)
			wordText[key] = strings.Join(ws, " ")
		}
		byWords[key] = append(byWords[key], p.id)
	}
	for _, key := range wordOrder {
		if ids := byWords[key]; len(ids) >= 2 {
			addAll(ids, "permuted", "same words in a different order: "+wordText[key])
		}
	}

	// near: a pair within the edit cap must share a 5-character head or tail and
	// differ in length by at most the cap, so only those buckets are compared.
	type nearTok struct{ id, tok string }
	near := map[int]nearTok{}
	buckets := map[string][]int{}
	var candidates [][2]int
	for ordinal, p := range named {
		if len(p.tok) < minPairLen {
			continue
		}
		prior := map[int]bool{}
		head, tail := p.tok[:minSharedAffix], p.tok[len(p.tok)-minSharedAffix:]
		for l := len(p.tok) - maxPairEdits; l <= len(p.tok)+maxPairEdits; l++ {
			for _, o := range buckets[fmt.Sprintf("h\x00%s\x00%d", head, l)] {
				prior[o] = true
			}
			for _, o := range buckets[fmt.Sprintf("t\x00%s\x00%d", tail, l)] {
				prior[o] = true
			}
		}
		for o := range prior {
			candidates = append(candidates, [2]int{o, ordinal})
		}
		near[ordinal] = nearTok{p.id, p.tok}
		hk := fmt.Sprintf("h\x00%s\x00%d", head, len(p.tok))
		tk := fmt.Sprintf("t\x00%s\x00%d", tail, len(p.tok))
		buckets[hk] = append(buckets[hk], ordinal)
		buckets[tk] = append(buckets[tk], ordinal)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i][0] != candidates[j][0] {
			return candidates[i][0] < candidates[j][0]
		}
		return candidates[i][1] < candidates[j][1]
	})
	for _, c := range candidates {
		a, b := near[c[0]], near[c[1]]
		if len(a.tok) > len(b.tok) {
			a, b = b, a
		}
		if a.id == b.id || a.tok == b.tok {
			continue
		}
		affix := sharedAffix(a.tok, b.tok)
		if affix < minSharedAffix {
			continue
		}
		if d := editDistanceWithin(a.tok, b.tok, maxPairEdits); d <= maxPairEdits {
			add(a.id, b.id, "near", fmt.Sprintf("%d edit(s) apart sharing %d characters: %s / %s", d, affix, a.tok, b.tok))
		}
	}

	out := make([]Pair, 0, len(found))
	for _, p := range found {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if pairRank[out[i].Kind] != pairRank[out[j].Kind] {
			return pairRank[out[i].Kind] < pairRank[out[j].Kind]
		}
		if out[i].A != out[j].A {
			return out[i].A < out[j].A
		}
		return out[i].B < out[j].B
	})
	return out
}

type sepReport struct {
	pairs       []GradedPair
	mutual      int
	oneSided    int
	undrawn     int
	edges       int
	mutualEdges int
	oneWay      []string
}

func byIDMap(rows []*Object) map[string]*Object {
	byid := map[string]*Object{}
	for _, r := range rows {
		if id := get(r, "id"); nonempty(id) {
			byid[id.(string)] = r
		}
	}
	return byid
}

func rowField(byid map[string]*Object, id, key string) Value {
	if r, ok := byid[id]; ok {
		return get(r, key)
	}
	return nil
}

func separationReport(byid map[string]*Object, edges edgeSet, pairs []Pair) sepReport {
	rep := sepReport{pairs: make([]GradedPair, 0, len(pairs)), oneWay: []string{}}
	for _, p := range pairs {
		ab, ba := edges.has(p.A, p.B), edges.has(p.B, p.A)
		state := "undrawn"
		switch {
		case ab && ba:
			state = "mutual"
			rep.mutual++
		case ab || ba:
			state = "one_sided"
			rep.oneSided++
		default:
			rep.undrawn++
		}
		rep.pairs = append(rep.pairs, GradedPair{
			A: p.A, B: p.B, Kind: p.Kind, Why: p.Why, State: state, AToB: ab, BToA: ba,
			ACanonical: rowField(byid, p.A, "canonical"), BCanonical: rowField(byid, p.B, "canonical"),
			AFamily: rowField(byid, p.A, "family"), BFamily: rowField(byid, p.B, "family"),
		})
	}
	for e := range edges {
		if edges.has(e[1], e[0]) {
			rep.mutualEdges++
		} else {
			rep.oneWay = append(rep.oneWay, e[0]+" -> "+e[1])
		}
	}
	sort.Strings(rep.oneWay)
	rep.edges = len(edges)
	return rep
}
