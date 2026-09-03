// Package selfquery lowers fak's existing self-description surfaces into one
// queryable FeatureCard catalog. It is a view over devindex, memq, gateway tool
// descriptors supplied by callers, and capindex capability cards where available.
package selfquery

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/capindex"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/memq"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

type Plane string

const (
	PlaneAll  Plane = "all"
	PlaneDev  Plane = "dev"
	PlaneLive Plane = "live"
)

type Effect string

const (
	EffectRead    Effect = "read"
	EffectPropose Effect = "propose"
	EffectMutate  Effect = "mutate"
)

type FeatureCard struct {
	Kind        string   `json:"kind"`
	Name        string   `json:"name"`
	Summary     string   `json:"summary"`
	Tags        []string `json:"tags,omitempty"`
	DetailRef   string   `json:"detail_ref"`
	Effect      Effect   `json:"effect"`
	RequiresCap string   `json:"requires_cap,omitempty"`
	Source      string   `json:"source"`
	Witness     string   `json:"witness"`
	// Root is the source repo root this card was loaded from (#3435, epic
	// #3434). Empty on the single-root path (Load/Query), so single-root JSON
	// and SummaryDigest stay byte-identical; LoadMany stamps it so a merged
	// cross-repo result stays attributable to the checkout it came from.
	Root string `json:"root,omitempty"`
	// Freshness is an advisory currency rung on a query-result card, distinct
	// from Witness (which proves WHICH bytes). Empty means not applicable /
	// unknown; otherwise FreshnessFresh, FreshnessSupersededPrefix+<name of the
	// superseding card>, or FreshnessStale (the cited artifact no longer exists on
	// disk). It is ranking metadata, NOT a trust verdict: a superseded or stale
	// card still returns, hedged with its rung, and freshness never re-orders or
	// drops a result. Computed only on the Query() path (see freshness.go);
	// Cards() leaves it empty so SummaryDigest stays stable.
	Freshness string       `json:"freshness,omitempty"`
	Request   RequestShape `json:"request"`
	Related   []Edge       `json:"related,omitempty"`
}

type RequestShape struct {
	Route     string         `json:"route"`
	MCPTool   string         `json:"mcp_tool,omitempty"`
	Command   []string       `json:"command,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`
	Note      string         `json:"note,omitempty"`
	Executed  bool           `json:"executed"`
}

type Detail struct {
	Card       FeatureCard     `json:"card"`
	Schema     json.RawMessage `json:"schema,omitempty"`
	Plan       *memq.Plan      `json:"plan,omitempty"`
	Query      *memq.Query     `json:"query,omitempty"`
	DocSnippet string          `json:"doc_snippet,omitempty"`
	CardBytes  string          `json:"card_bytes,omitempty"`
}

type Response struct {
	Root           string             `json:"root,omitempty"`
	Query          string             `json:"query"`
	Plane          Plane              `json:"plane"`
	Cards          []FeatureCard      `json:"cards"`
	Detail         *Detail            `json:"detail,omitempty"`
	Clarifications *ClarificationPlan `json:"clarifications,omitempty"`
}

const DefaultResultLimit = 5

type Request struct {
	Root           string
	Query          string
	Plane          Plane
	Detail         string
	Limit          int
	All            bool
	MissingContext []string
}

type ToolDescriptor struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// DevLeaf, DevDoc, DevClaim, and DevVerb are an optional, data-only repository
// catalog supplied by fak-dev. Keeping these records here preserves the shared query
// engine without importing the repository-development index into the runtime graph.
type DevLeaf struct{ Name, Tree, Desc string }
type DevDoc struct{ Title, Path, Blurb string }
type DevClaim struct {
	Tag, Text string
	Lanes     []string
}
type DevVerb struct {
	Name, Synopsis, Lane string
	Aliases              []string
}
type DevCatalog struct {
	Leaves []DevLeaf
	Docs   []DevDoc
	Claims []DevClaim
	Verbs  []DevVerb
}

// DevLoader is the optional repository-development seam. Runtime callers leave it nil;
// fak-dev installs a loader before querying. The function type avoids any runtime import
// of internal/devindex while preserving the existing data-rich query behavior.
type DevLoader func(root string) (*DevCatalog, error)

type Options struct {
	Tools     []ToolDescriptor
	CapCards  []capindex.CapCard
	Dev       *DevCatalog
	DevLoader DevLoader
}

type Catalog struct {
	root     string
	dev      *DevCatalog
	tools    []ToolDescriptor
	capCards []capindex.CapCard
	caps     *capindex.Catalog
}

func Load(root string, opt Options) (*Catalog, error) {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	c := &Catalog{root: root, dev: opt.Dev, tools: append([]ToolDescriptor(nil), opt.Tools...)}
	if c.dev == nil && opt.DevLoader != nil {
		dev, err := opt.DevLoader(root)
		if err != nil && len(opt.Tools) == 0 && len(opt.CapCards) == 0 {
			return nil, err
		}
		c.dev = dev
	}
	c.capCards = append(loadRootCapCards(root, &c.caps), opt.CapCards...)
	sort.Slice(c.tools, func(i, j int) bool { return c.tools[i].Name < c.tools[j].Name })
	sort.Slice(c.capCards, func(i, j int) bool {
		if c.capCards[i].Ref.Kind != c.capCards[j].Ref.Kind {
			return c.capCards[i].Ref.Kind < c.capCards[j].Ref.Kind
		}
		return c.capCards[i].Ref.Name < c.capCards[j].Ref.Name
	})
	return c, nil
}

func loadRootCapCards(root string, out **capindex.Catalog) []capindex.CapCard {
	if strings.TrimSpace(root) == "" {
		return nil
	}
	skills := filepath.Join(root, ".claude", "skills")
	if info, err := os.Stat(skills); err != nil || !info.IsDir() {
		return nil
	}
	cat := capindex.NewCatalog()
	resolver := capindex.NewSkillResolver(skills)
	cards := resolver.Index()
	cat.AddResolver(capindex.CapKindSkill, resolver)
	cat.Sync()
	*out = cat
	return cards
}

func ToolDescriptorsFromMaps(in []map[string]any) []ToolDescriptor {
	out := make([]ToolDescriptor, 0, len(in))
	for _, td := range in {
		name, _ := td["name"].(string)
		desc, _ := td["description"].(string)
		if strings.TrimSpace(name) == "" {
			continue
		}
		var schema json.RawMessage
		switch v := td["inputSchema"].(type) {
		case json.RawMessage:
			schema = append(json.RawMessage(nil), v...)
		case []byte:
			schema = append(json.RawMessage(nil), v...)
		case string:
			schema = json.RawMessage(v)
		case nil:
		default:
			if b, err := json.Marshal(v); err == nil {
				schema = b
			}
		}
		out = append(out, ToolDescriptor{Name: name, Description: desc, InputSchema: schema})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (c *Catalog) Query(req Request) (Response, error) {
	return runQuery(req, queryPaths{
		root:  c.root,
		cards: c.Cards,
		// Advisory currency rungs (#3163) for a SINGLE root: two independent axes are
		// merged — supersession (a newer same-topic dated note supersedes an older one)
		// and staleness (a card whose cited artifact was deleted since authoring).
		// STALE wins when both apply: a removed artifact's supersession is moot.
		rungs: func(all []FeatureCard) map[string]string {
			rungs := freshnessByKey(all)
			for k, v := range stalenessByKey(all, c.root) {
				rungs[k] = v
			}
			return rungs
		},
		apply:   applyFreshness,
		rungKey: cardKey,
		detail:  c.detail,
	})
}

// queryPaths names everything the single-root and the cross-repo query paths do
// DIFFERENTLY. Everything else about a feature query — validating the request,
// building the candidate set, ranking, the limit cut, the clarification plan and the
// faulted-detail lookup — is identical, which is what runQuery is. MultiCatalog.Query's
// own doc comment promises it "mirrors Catalog.Query so the single-root and multi-root
// paths behave identically apart from the fan-out"; this makes that one body instead of
// two that have to be kept in step by hand.
type queryPaths struct {
	// root is the Root stamped on the response — the catalog's own root, or the
	// multi-catalog's primary.
	root string
	// cards returns the candidate set for a plane (one catalog's, or the Root-stamped
	// union of every member's).
	cards func(Plane) []FeatureCard
	// rungs computes the advisory currency rungs over the FULL candidate set. The two
	// paths merge the same two axes but at different scope: the single-root path resolves
	// staleness against its one checkout, while the multi path scopes supersession and
	// staleness per source root and returns root-qualified keys.
	rungs func([]FeatureCard) map[string]string
	// apply stamps the rungs onto the ranked cards, using the same key scheme as rungKey.
	apply func([]FeatureCard, map[string]string)
	// rungKey indexes one card into the rungs map: the bare cardKey on the single-root
	// path, root-qualified on the multi path so two repos carrying a same-named card
	// never cross-contaminate rungs.
	rungKey func(FeatureCard) string
	// detail builds the faulted detail from the card's OWNING catalog — the one member
	// whose checkout the card's DetailRef resolves against.
	detail func(FeatureCard, string) (Detail, error)
}

// runQuery is the shared body of Catalog.Query and MultiCatalog.Query. Freshness is
// advisory throughout: it is stamped onto the ranked result and the faulted detail card,
// and never re-orders or drops one.
func runQuery(req Request, p queryPaths) (Response, error) {
	q := strings.TrimSpace(req.Query)
	if q == "" {
		return Response{}, errors.New("feature query requires a non-empty query")
	}
	if req.Limit < 0 {
		return Response{}, errors.New("feature query limit must be non-negative")
	}
	plane := normalizePlane(req.Plane)
	if plane == "" {
		return Response{}, fmt.Errorf("unknown feature query plane %q (want dev, live, or all)", req.Plane)
	}
	all := p.cards(plane)
	cards := rankCards(all, q)
	limit := req.Limit
	if limit == 0 && !req.All {
		limit = DefaultResultLimit
	}
	if limit > 0 && len(cards) > limit {
		cards = cards[:limit]
	}
	rungs := p.rungs(all)
	p.apply(cards, rungs)
	resp := Response{Root: p.root, Query: q, Plane: plane, Cards: cards}
	if len(req.MissingContext) > 0 {
		plan := MissingContextClarifications(req.MissingContext)
		resp.Clarifications = &plan
	}
	if strings.TrimSpace(req.Detail) != "" {
		card, ok := findCard(all, req.Detail)
		if !ok {
			return Response{}, fmt.Errorf("feature detail %q not found", req.Detail)
		}
		if r, ok := rungs[p.rungKey(card)]; ok {
			card.Freshness = r
		}
		d, err := p.detail(card, q)
		if err != nil {
			return Response{}, err
		}
		resp.Detail = &d
	}
	return resp, nil
}

func (c *Catalog) Cards(plane Plane) []FeatureCard {
	plane = normalizePlane(plane)
	var out []FeatureCard
	if plane == PlaneAll || plane == PlaneDev {
		out = append(out, c.devCards()...)
	}
	if plane == PlaneAll || plane == PlaneLive {
		out = append(out, c.contextPlanCards()...)
		out = append(out, c.askPolicyCards()...)
		out = append(out, c.memoryCards()...)
		out = append(out, c.storeCards()...)
		out = append(out, c.toolCards()...)
		out = append(out, c.capabilityCards()...)
	}
	sortCards(out)
	return out
}

func (c *Catalog) SummaryDigest() string {
	b, _ := json.Marshal(c.Cards(PlaneAll))
	return capindex.Digest(b)
}

func (c *Catalog) Sources() []string {
	src := map[string]bool{}
	for _, card := range c.Cards(PlaneAll) {
		src[card.Source] = true
	}
	out := make([]string, 0, len(src))
	for s := range src {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

func (c *Catalog) devCards() []FeatureCard {
	if c.dev == nil {
		return nil
	}
	var out []FeatureCard
	out = append(out, c.devSurfaceCards()...)
	for _, l := range c.dev.Leaves {
		summary := l.Desc
		if summary == "" {
			summary = l.Tree
		}
		out = append(out, card("dev-leaf", "leaf:"+l.Name, summary,
			[]string{"dev", "leaf", "lane", l.Name, "commit", "stamp"},
			"fak-dev index leaf "+l.Name, EffectRead, "", "devindex", digestOf(l),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "leaf", l.Name}, Executed: false}))
	}
	for _, d := range c.dev.Docs {
		out = append(out, card("dev-doc", "doc:"+d.Title, strmatch.FirstTrimmed(d.Blurb, d.Path),
			[]string{"dev", "doc", "docs", "index", d.Path},
			d.Path, EffectRead, "", "devindex", digestOf(d),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "docs", d.Title}, Executed: false}))
	}
	for _, cl := range c.dev.Claims {
		out = append(out, card("dev-claim", "claim:"+shortName(cl.Text), cl.Text,
			append([]string{"dev", "claim", "claims", strings.ToLower(cl.Tag)}, cl.Lanes...),
			"fak-dev index claims "+strings.Join(cl.Lanes, " "), EffectRead, "", "devindex", digestOf(cl),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "claims", strings.Join(cl.Lanes, " ")}, Executed: false}))
	}
	for _, v := range c.dev.Verbs {
		tags := append([]string{"dev", "cli", "verb", v.Lane}, v.Aliases...)
		out = append(out, card("cli-verb", "fak-dev "+v.Name, v.Synopsis, tags,
			"fak-dev "+v.Name, EffectRead, "", "devindex", digestOf(v),
			RequestShape{Route: "cli", Command: []string{"fak-dev", v.Name, "--help"}, Executed: false}))
	}
	return out
}

func (c *Catalog) devSurfaceCards() []FeatureCard {
	return []FeatureCard{
		card("dev-query", "fak-dev index lane", "resolve a path to its owning lane and suggested commit stamp",
			[]string{"dev", "index", "lane", "commit", "stamp", "path", "owner"},
			"fak-dev index lane <path>", EffectRead, "", "devindex", digestOf("index-lane"),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "lane", "<path>"}, Executed: false}),
		card("dev-query", "fak-dev index docs", "search the curated INDEX.md doc map by query",
			[]string{"dev", "index", "docs", "doc", "documentation"},
			"fak-dev index docs <query>", EffectRead, "", "devindex", digestOf("index-docs"),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "docs", "<query>"}, Executed: false}),
		card("dev-query", "fak-dev index claims", "search the CLAIMS.md honesty ledger by capability, lane, or token",
			[]string{"dev", "index", "claims", "shipped", "simulated", "stub"},
			"fak-dev index claims <query>", EffectRead, "", "devindex", digestOf("index-claims"),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "claims", "<query>"}, Executed: false}),
		card("dev-query", "fak-dev index verbs", "search fak's live CLI verb catalog",
			[]string{"dev", "index", "verbs", "cli", "command"},
			"fak-dev index verbs <query>", EffectRead, "", "devindex", digestOf("index-verbs"),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "verbs", "<query>"}, Executed: false}),
		card("dev-query", "fak-dev index work", "list or search the repo's declared issue views and default dispatchable backlog",
			[]string{"dev", "index", "work", "issue", "issues", "backlog", "dispatch", "ready"},
			"fak-dev index work [<query>]", EffectRead, "", "devindex", digestOf("index-work"),
			RequestShape{Route: "cli", Command: []string{"fak-dev", "index", "work", "<query>"}, Executed: false}),
	}
}

func (c *Catalog) contextPlanCards() []FeatureCard {
	sources := []ctxplan.AssumptionSource{
		ctxplan.AssumptionUserStated,
		ctxplan.AssumptionWitnessed,
		ctxplan.AssumptionInferred,
		ctxplan.AssumptionStale,
		ctxplan.AssumptionUnknown,
	}
	tags := []string{"live", "managed-context", "context", "plan", "assumption", "assumptions", "confidence", "query", "refresh"}
	for _, source := range sources {
		tags = append(tags, string(source))
	}
	return []FeatureCard{
		card("context-plan", "context-plan:assumptions",
			"score context assumptions by source class and confidence before effect use",
			tags,
			"internal/ctxplan/assumption.go", EffectRead, "", "ctxplan", digestOf(sources),
			RequestShape{
				Route: "library",
				Note:  "call ctxplan.AssessAssumptions or supply PlanQuery.Assumptions; low-confidence, stale, and unknown assumptions return query/refresh actions before effects",
			}),
	}
}

func (c *Catalog) askPolicyCards() []FeatureCard {
	tags := []string{
		"live", "managed-context", "ask", "assume", "assumption", "clarification",
		"policy", "stakes", "reversibility", "confidence", "threshold",
	}
	return []FeatureCard{
		card("ask-policy", "ask-policy:should-ask",
			"decide whether fak asks the user instead of silently assuming, by stakes, reversibility, and confidence",
			tags,
			"internal/selfquery/ask_policy.go", EffectRead, "", "selfquery", digestOf("ask-policy:should-ask"),
			RequestShape{
				Route: "library",
				Note:  "call selfquery.ShouldAsk(AskInput{Confidence, Stakes, Reversibility}); below the applicable threshold it returns ShouldAsk=true instead of assuming",
			}),
	}
}

func (c *Catalog) memoryCards() []FeatureCard {
	ds := memq.Drivers()
	out := make([]FeatureCard, 0, len(ds))
	for _, d := range ds {
		q := d.Build(memq.Params{Intent: "the task at hand"})
		p := memq.Explain(q)
		effect, cap := memoryEffect(p)
		out = append(out, card("memory-driver", "memory-driver:"+d.Name, d.Doc,
			[]string{"live", "memory", "memq", "driver", d.Name},
			"memq:"+d.Name, effect, cap, "memq", digestOf(struct {
				Name string
				Doc  string
				Plan memq.Plan
			}{d.Name, d.Doc, p}),
			RequestShape{
				Route:   "memory-explain-before-run",
				MCPTool: "fak_memory_explain",
				Command: []string{"fak", "memory", "explain", "--driver", d.Name},
				Arguments: map[string]any{
					"driver": d.Name,
				},
				Note:     "explain first; fak_memory_run defaults to apply=false so mutations are proposed unless explicitly authorized",
				Executed: false,
			}))
	}
	return out
}

func (c *Catalog) storeCards() []FeatureCard {
	drivers := session.StoreDrivers()
	out := make([]FeatureCard, 0, len(drivers))
	for _, name := range drivers {
		out = append(out, card("session-store-driver", "session-store:"+name,
			fmt.Sprintf("durable session persistence driver %q", name),
			[]string{"session", "store", "driver", name},
			"internal/session", EffectRead, "session-store", "session",
			digestOf(name),
			RequestShape{
				Route: "session-store",
				Note:  "registered session store driver",
			}))
	}
	return out
}

func (c *Catalog) toolCards() []FeatureCard {
	out := make([]FeatureCard, 0, len(c.tools))
	for _, td := range c.tools {
		effect, cap := toolEffect(td.Name)
		out = append(out, card("mcp-tool", td.Name, td.Description,
			toolTags(td.Name), "mcp:"+td.Name, effect, cap, "gateway.tools", digestOf(td),
			toolRequest(td.Name, effect)))
	}
	return out
}

func (c *Catalog) capabilityCards() []FeatureCard {
	out := make([]FeatureCard, 0, len(c.capCards))
	for _, cc := range c.capCards {
		name := string(cc.Ref.Kind) + ":" + cc.Ref.Name
		out = append(out, card("capability", name, cc.Trigger,
			append([]string{"live", "capability", string(cc.Ref.Kind)}, cc.Tags...),
			capRefString(cc.Ref), EffectRead, "", "capindex", strmatch.FirstTrimmed(cc.Digest, digestOf(cc)),
			RequestShape{
				Route: "fault-capability-detail",
				Note:  "fault the selected capability body through its resolver before use",
			}))
	}
	return out
}

func (c *Catalog) detail(card FeatureCard, intent string) (Detail, error) {
	d := Detail{Card: card}
	switch card.Kind {
	case "mcp-tool":
		for _, td := range c.tools {
			if td.Name == card.Name {
				d.Schema = append(json.RawMessage(nil), td.InputSchema...)
				return d, nil
			}
		}
	case "memory-driver":
		name := strings.TrimPrefix(card.Name, "memory-driver:")
		driver, ok := memq.Get(name)
		if !ok {
			return d, fmt.Errorf("memory driver %q no longer registered", name)
		}
		q := driver.Build(memq.Params{Intent: intent})
		p := memq.Explain(q)
		d.Query = &q
		d.Plan = &p
		return d, nil
	case "dev-doc":
		snip, err := c.docSnippet(card.DetailRef)
		if err != nil {
			return d, err
		}
		d.DocSnippet = snip
		return d, nil
	case "capability":
		for _, cc := range c.capCards {
			if capRefString(cc.Ref) == card.DetailRef {
				if c.caps != nil {
					if cap, err := c.caps.Lookup(cc.Ref); err == nil {
						if body := cap.Materialize(); len(body) > 0 {
							d.CardBytes = string(body)
							return d, nil
						}
					}
				}
				d.CardBytes = string(cc.CardBytes)
				return d, nil
			}
		}
	}
	return d, nil
}

func (c *Catalog) docSnippet(ref string) (string, error) {
	if ref == "" || strings.HasPrefix(ref, "http://") || strings.HasPrefix(ref, "https://") {
		return "", nil
	}
	clean := ref
	if i := strings.IndexAny(clean, "#?"); i >= 0 {
		clean = clean[:i]
	}
	if clean == "" {
		return "", nil
	}
	b, err := os.ReadFile(filepath.Join(c.root, filepath.FromSlash(clean)))
	if err != nil {
		return "", nil
	}
	s := string(b)
	r := []rune(s)
	if len(r) > 1200 {
		return string(r[:1200]), nil
	}
	return s, nil
}

func card(kind, name, summary string, tags []string, detailRef string, effect Effect, reqCap, source, witness string, req RequestShape) FeatureCard {
	return FeatureCard{
		Kind:        kind,
		Name:        name,
		Summary:     strings.TrimSpace(summary),
		Tags:        cleanTags(tags),
		DetailRef:   detailRef,
		Effect:      effect,
		RequiresCap: reqCap,
		Source:      source,
		Witness:     witness,
		Request:     req,
	}
}

func memoryEffect(p memq.Plan) (Effect, string) {
	if len(p.Mutations) == 0 {
		return EffectRead, ""
	}
	return EffectPropose, strings.Join(p.Mutations, ",")
}

func toolEffect(name string) (Effect, string) {
	switch name {
	case "fak_memory_run":
		return EffectPropose, "memq.apply"
	case "fak_adjudicate", "fak_read", "fak_memory_drivers", "fak_memory_explain", "fak_tools_search",
		"fak_feature_query", "fak_capabilities":
		return EffectRead, ""
	default:
		if strings.Contains(name, "run") || strings.Contains(name, "syscall") || strings.Contains(name, "admit") ||
			strings.Contains(name, "revoke") || strings.Contains(name, "reset") || strings.Contains(name, "change") {
			return EffectMutate, "tool-call"
		}
		return EffectRead, ""
	}
}

func toolRequest(name string, effect Effect) RequestShape {
	if strings.HasPrefix(name, "fak_memory_") {
		args := map[string]any{}
		if name == "fak_memory_explain" || name == "fak_memory_run" {
			args["driver"] = "<driver>"
			if name == "fak_memory_run" {
				args["apply"] = false
			}
		}
		return RequestShape{Route: "mcp/tools-call", MCPTool: name, Arguments: args, Note: "memory mutations stay proposal-only unless apply=true is explicitly authorized", Executed: false}
	}
	if strings.HasPrefix(name, "fak_") {
		return RequestShape{Route: "mcp/tools-call", MCPTool: name, Arguments: map[string]any{}, Executed: false}
	}
	return RequestShape{
		Route:   "adjudicate-before-execute",
		MCPTool: "fak_adjudicate",
		Arguments: map[string]any{
			"tool":      name,
			"arguments": map[string]any{},
		},
		Note:     "discovery does not execute this tool; call fak_adjudicate or fak_syscall with concrete arguments",
		Executed: false,
	}
}

func toolTags(name string) []string {
	tags := []string{"live", "mcp", "tool"}
	for _, part := range strings.FieldsFunc(name, func(r rune) bool { return r == '_' || r == '-' }) {
		if part != "" {
			tags = append(tags, part)
		}
	}
	return tags
}

// rankCards ranks a candidate card set against a query. As of #3235 (epic
// #3229) it delegates to the hybrid BM25 + stemming-expansion retriever in
// ranker.go, replacing the flat lexical token-overlap scorer (score, below).
// The signature and the deterministic cardLess tiebreak are unchanged, so every
// caller (Query, Capabilities, the freshness ordering check) is byte-compatible.
func rankCards(cards []FeatureCard, q string) []FeatureCard {
	return rankHybrid(cards, q)
}

// score is the pre-#3235 flat lexical scorer: fixed per-field token-overlap
// weights, no IDF, no stemming. A term common to every live card
// (live/mcp/tool) scores here exactly like a discriminating one — the failure
// mode ranker.go's IDF rung fixes. It is retained OFF the production path as the
// A/B baseline the recall regression test (ranker_test.go) grades the hybrid
// ranker against.
func score(c FeatureCard, toks []string) int {
	name := strings.ToLower(c.Name)
	summary := strings.ToLower(c.Summary)
	tags := strings.ToLower(strings.Join(c.Tags, " "))
	ref := strings.ToLower(c.DetailRef + " " + c.Kind + " " + c.Source)
	total := 0
	for _, tk := range toks {
		if name == tk || strings.HasSuffix(name, ":"+tk) || strings.HasSuffix(name, " "+tk) {
			total += 12
		}
		for _, tag := range c.Tags {
			if tag == tk {
				total += 12
				break
			}
		}
		if strings.Contains(name, tk) {
			total += 5
		}
		if strings.Contains(tags, tk) {
			total += 4
		}
		if strings.Contains(ref, tk) {
			total += 2
		}
		if strings.Contains(summary, tk) {
			total++
		}
	}
	return total
}

func findCard(cards []FeatureCard, key string) (FeatureCard, bool) {
	k := normalizeKey(key)
	for _, c := range cards {
		if normalizeKey(c.Name) == k || normalizeKey(c.DetailRef) == k {
			return c, true
		}
	}
	for _, c := range cards {
		if strings.Contains(normalizeKey(c.Name), k) {
			return c, true
		}
	}
	return FeatureCard{}, false
}

func normalizePlane(p Plane) Plane {
	switch strings.ToLower(strings.TrimSpace(string(p))) {
	case "", "all":
		return PlaneAll
	case "dev":
		return PlaneDev
	case "live":
		return PlaneLive
	default:
		return ""
	}
}

func sortCards(cards []FeatureCard) {
	sort.SliceStable(cards, func(i, j int) bool { return cardLess(cards[i], cards[j]) })
}

func cardLess(a, b FeatureCard) bool {
	if a.Source != b.Source {
		return a.Source < b.Source
	}
	if a.Kind != b.Kind {
		return a.Kind < b.Kind
	}
	if a.Name != b.Name {
		return a.Name < b.Name
	}
	// Root is the final tiebreak so a cross-repo merge (#3435) is deterministic
	// when two checkouts carry a same-named card. Back-compat: single-root cards
	// all have Root=="" so this branch never fires on the single-root path.
	return a.Root < b.Root
}

func cleanTags(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, t := range in {
		t = strings.ToLower(strings.TrimSpace(t))
		if t == "" || seen[t] {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

func tokens(q string) []string {
	fields := strings.FieldsFunc(strings.ToLower(q), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
	var out []string
	for _, f := range fields {
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}

func digestOf(v any) string {
	b, _ := json.Marshal(v)
	return capindex.Digest(b)
}

func capRefString(ref capindex.CapRef) string {
	if ref.Version == "" {
		return string(ref.Kind) + ":" + ref.Name
	}
	return string(ref.Kind) + ":" + ref.Name + "@" + ref.Version
}

func normalizeKey(s string) string {
	return strings.Join(tokens(s), " ")
}

func shortName(s string) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) > 48 {
		r = r[:48]
	}
	return strings.TrimSpace(string(r))
}
