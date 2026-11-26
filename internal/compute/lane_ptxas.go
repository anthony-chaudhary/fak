package compute

// lane_ptxas.go ingests the diagnostic stream of `nvcc -Xptxas -v` into per-kernel facts (registers,
// static shared memory, spill traffic), joins those facts to the static launch inventory, and prices each
// joined launch on an NVArch row through the lane ledger. ptxas reports a clause only when it is non-zero,
// so an absent smem or spill clause is recorded as zero with its Has flag left false, while a missing
// "Used N registers" clause marks the entry incomplete rather than zero registers. A clause this parser does
// not recognise is kept verbatim and counted; it never fails the parse.

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

const (
	ptxasProvenance         = "registers parsed from caller-supplied ptxas log"
	ptxasMaxLineBytes       = 16 << 20
	ptxasBuiltinCodes       = "vwbcahstijlmxynofdegz"
	ptxasAnonymousNamespace = "_GLOBAL__N_1"
	ptxasMaxSeqDigits       = 6
	ptxasIntegerCodes       = "ahcstijlmxy"
	ptxasTypeArg            = "?"
)

// Each refusal is a distinct sentinel so a caller can branch on the cause with errors.Is.
var (
	ErrPtxasNoArch          = errors.New("ptxas sweep: the log names no sm_ architecture to check against the NVArch label")
	ErrPtxasArchMismatch    = errors.New("ptxas sweep: the log's architecture does not match the NVArch label")
	ErrPtxasDeviceSMs       = errors.New("ptxas sweep: deviceSMs must be positive; the SM count is a caller input")
	ErrPtxasGridBlocks      = errors.New("ptxas sweep: gridBlocks must be positive; the grid is a caller input")
	ErrPtxasIncomplete      = errors.New("ptxas entry has no Used registers line, so its register count is unmeasured")
	ErrPtxasSymbolicThreads = errors.New("ptxas sweep: the site's ThreadsPerBlock is symbolic, so the launch cannot be priced")
)

// PtxasEntry is one compiled entry function as ptxas reported it. BaseName is empty when the mangled name
// is outside the demangled subset; such an entry can never join a launch site.
type PtxasEntry struct {
	Mangled              string        `json:"mangled"`
	BaseName             string        `json:"base_name"`
	TemplateArgs         []string      `json:"template_args,omitempty"`
	Arch                 string        `json:"arch"`
	Line                 int           `json:"line"`
	HasRegisters         bool          `json:"has_registers"`
	Registers            int           `json:"registers"`
	Barriers             int           `json:"barriers"`
	SmemBytes            int64         `json:"smem_bytes"`
	CumulativeStackBytes int64         `json:"cumulative_stack_bytes"`
	CmemBytes            map[int]int64 `json:"cmem_bytes,omitempty"`
	HasFrame             bool          `json:"has_frame"`
	StackFrameBytes      int64         `json:"stack_frame_bytes"`
	SpillStoreBytes      int64         `json:"spill_store_bytes"`
	SpillLoadBytes       int64         `json:"spill_load_bytes"`
	Unparsed             []string      `json:"unparsed,omitempty"`
}

// PtxasUnitExit is one compilation unit whose nvcc run exited non-zero. Its entries, if any, are still
// reported; the failure says the stream may be partial.
type PtxasUnitExit struct {
	Unit string `json:"unit"`
	Exit int    `json:"exit"`
}

// PtxasCounts tallies what the parse saw. Every count is a number of lines, units or entries, never a sum
// of unlike units.
type PtxasCounts struct {
	Entries         int `json:"entries"`
	Incomplete      int `json:"incomplete"`
	Undemangled     int `json:"undemangled"`
	UnparsedClauses int `json:"unparsed_clauses"`
	CalleeBlocks    int `json:"callee_blocks"`
	ToolErrorLines  int `json:"tool_error_lines"`
	Units           int `json:"units"`
	FailedUnits     int `json:"failed_units"`
}

// PtxasReport is the parsed diagnostic stream of one nvcc run.
type PtxasReport struct {
	Entries     []PtxasEntry    `json:"entries"`
	ArchsSeen   []string        `json:"archs_seen"`
	Unattached  []string        `json:"unattached_unparsed,omitempty"`
	FailedUnits []PtxasUnitExit `json:"failed_units,omitempty"`
	Counts      PtxasCounts     `json:"counts"`
}

// ParsePtxasLog reads an `nvcc -Xptxas -v` stream. The only error is a read failure, including a line
// longer than the reader's limit, so a truncated stream is never silently accepted.
func ParsePtxasLog(r io.Reader) (PtxasReport, error) {
	p := ptxasParser{archs: map[string]bool{}}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64<<10), ptxasMaxLineBytes)
	for sc.Scan() {
		p.lineNo++
		p.line(strings.TrimRight(sc.Text(), "\r"))
	}
	if err := sc.Err(); err != nil {
		return PtxasReport{}, fmt.Errorf("ptxas log after line %d: %w", p.lineNo, err)
	}
	return p.finish(), nil
}

type ptxasParser struct {
	report      PtxasReport
	lineNo      int
	hasCur      bool
	cur         int
	expectFrame bool
	usedSeen    bool
	archs       map[string]bool
}

func (p *ptxasParser) line(text string) {
	if body, ok := ptxasInfoBody(text); ok {
		p.info(body, text)
		return
	}
	if rest, ok := strings.CutPrefix(text, "##### UNIT "); ok {
		p.unit(rest, text)
		return
	}
	if strings.HasPrefix(text, "ptxas ") {
		p.report.Counts.ToolErrorLines++
		return
	}
	if strings.Contains(text, "bytes stack frame") {
		p.frame(text)
	}
}

// unit reads the "<name> exit=<n>" tail of a unit marker. A marker of another shape is kept verbatim.
func (p *ptxasParser) unit(rest, raw string) {
	fields := strings.Fields(rest)
	if len(fields) != 2 {
		p.report.Unattached = append(p.report.Unattached, raw)
		return
	}
	code, ok := strings.CutPrefix(fields[1], "exit=")
	exit, err := strconv.Atoi(code)
	if !ok || err != nil {
		p.report.Unattached = append(p.report.Unattached, raw)
		return
	}
	p.report.Counts.Units++
	if exit != 0 {
		p.report.FailedUnits = append(p.report.FailedUnits, PtxasUnitExit{Unit: fields[0], Exit: exit})
		p.report.Counts.FailedUnits++
	}
}

// ptxasInfoBody returns the text after "ptxas info ...: " for an info line.
func ptxasInfoBody(text string) (string, bool) {
	rest, ok := strings.CutPrefix(text, "ptxas info")
	if !ok {
		return "", false
	}
	_, body, found := strings.Cut(rest, ": ")
	if !found {
		return "", true
	}
	return strings.TrimSpace(body), true
}

func (p *ptxasParser) info(body, raw string) {
	switch {
	case strings.HasPrefix(body, "Compiling entry function "):
		p.compiling(strings.TrimPrefix(body, "Compiling entry function "), raw)
	case strings.HasPrefix(body, "Function properties for "):
		p.properties(strings.TrimPrefix(body, "Function properties for "))
	case strings.HasPrefix(body, "Used "):
		p.used(strings.TrimPrefix(body, "Used "), raw)
	case strings.HasPrefix(body, "Compile time = "), strings.HasSuffix(body, " bytes gmem"):
		// Known lines that describe no entry's registers, shared memory or spills.
	default:
		p.unparsed(raw)
	}
}

func (p *ptxasParser) compiling(rest, raw string) {
	mangled, arch, ok := ptxasParseCompiling(rest)
	if !ok {
		p.hasCur = false
		p.unparsed(raw)
		return
	}
	base, args, demangled := ptxasDemangle(mangled)
	if !demangled {
		base, args = "", nil
	}
	p.report.Entries = append(p.report.Entries, PtxasEntry{
		Mangled: mangled, BaseName: base, TemplateArgs: args, Arch: arch, Line: p.lineNo,
	})
	p.cur, p.hasCur = len(p.report.Entries)-1, true
	p.expectFrame, p.usedSeen = false, false
	p.archs[arch] = true
}

// ptxasParseCompiling reads the body of "Compiling entry function '<name>' for 'sm_N'".
func ptxasParseCompiling(rest string) (string, string, bool) {
	rest, ok := strings.CutPrefix(rest, "'")
	if !ok {
		return "", "", false
	}
	name, after, found := strings.Cut(rest, "' for '")
	if !found || name == "" {
		return "", "", false
	}
	arch, ok := strings.CutSuffix(after, "'")
	if !ok || !strings.HasPrefix(arch, "sm_") {
		return "", "", false
	}
	return name, arch, true
}

// properties marks the next stack-frame line as this entry's only when the block names the current entry.
// Callee blocks (ptxas-internal helpers printed before an entry) have no Used line and are only counted.
func (p *ptxasParser) properties(name string) {
	if p.hasCur && name == p.report.Entries[p.cur].Mangled {
		p.expectFrame = true
		return
	}
	p.expectFrame = false
	p.report.Counts.CalleeBlocks++
}

func (p *ptxasParser) frame(text string) {
	if !p.expectFrame {
		return
	}
	p.expectFrame = false
	e := &p.report.Entries[p.cur]
	f, ok := ptxasParseFrame(strings.TrimSpace(text))
	if !ok {
		e.Unparsed = append(e.Unparsed, text)
		return
	}
	e.HasFrame = true
	e.StackFrameBytes, e.SpillStoreBytes, e.SpillLoadBytes = f.stack, f.store, f.load
}

type ptxasFrame struct{ stack, store, load int64 }

// ptxasParseFrame reads "<n> bytes stack frame, <n> bytes spill stores, <n> bytes spill loads".
func ptxasParseFrame(s string) (ptxasFrame, bool) {
	parts := strings.Split(s, ", ")
	if len(parts) != 3 {
		return ptxasFrame{}, false
	}
	stack, ok1 := ptxasCountWithUnit(parts[0], " bytes stack frame")
	store, ok2 := ptxasCountWithUnit(parts[1], " bytes spill stores")
	load, ok3 := ptxasCountWithUnit(parts[2], " bytes spill loads")
	return ptxasFrame{stack, store, load}, ok1 && ok2 && ok3
}

func (p *ptxasParser) used(body, raw string) {
	if !p.hasCur {
		p.unparsed(raw)
		return
	}
	e := &p.report.Entries[p.cur]
	if p.usedSeen {
		e.Unparsed = append(e.Unparsed, raw)
		return
	}
	p.usedSeen = true
	seen := map[string]bool{}
	for _, clause := range strings.Split(body, ", ") {
		kind, ok := ptxasApplyUsedClause(&PtxasEntry{}, clause)
		if !ok || seen[kind] {
			e.Unparsed = append(e.Unparsed, clause)
			continue
		}
		seen[kind] = true
		ptxasApplyUsedClause(e, clause)
	}
}

// ptxasApplyUsedClause writes one clause of a Used line into e and returns the clause kind, so a repeated
// kind is refused as unparsed. Clause order is not significant: ptxas has printed the stack-size clause
// before the smem clause, and the grammar does not depend on that order.
func ptxasApplyUsedClause(e *PtxasEntry, clause string) (string, bool) {
	if n, ok := ptxasCountWithUnit(clause, " registers"); ok {
		e.Registers, e.HasRegisters = int(n), true
		return "registers", true
	}
	if rest, ok := strings.CutPrefix(clause, "used "); ok {
		n, ok := ptxasCountWithUnit(rest, " barriers")
		e.Barriers = int(n)
		return "barriers", ok
	}
	if n, ok := ptxasCountWithUnit(clause, " bytes smem"); ok {
		e.SmemBytes = n
		return "smem", true
	}
	if n, ok := ptxasCountWithUnit(clause, " bytes cumulative stack size"); ok {
		e.CumulativeStackBytes = n
		return "cumulative_stack", true
	}
	return ptxasApplyCmemClause(e, clause)
}

// ptxasApplyCmemClause reads "<n> bytes cmem[<bank>]".
func ptxasApplyCmemClause(e *PtxasEntry, clause string) (string, bool) {
	head, tail, found := strings.Cut(clause, " bytes cmem[")
	if !found {
		return "", false
	}
	bankText, ok := strings.CutSuffix(tail, "]")
	if !ok {
		return "", false
	}
	bank, okBank := ptxasParseCount(bankText)
	n, okNum := ptxasParseCount(head)
	if !okBank || !okNum {
		return "", false
	}
	if e.CmemBytes == nil {
		e.CmemBytes = map[int]int64{}
	}
	e.CmemBytes[int(bank)] = n
	return "cmem[" + bankText + "]", true
}

// ptxasCountWithUnit reads "<digits><unit>" where unit includes its leading space.
func ptxasCountWithUnit(s, unit string) (int64, bool) {
	num, ok := strings.CutSuffix(s, unit)
	if !ok {
		return 0, false
	}
	return ptxasParseCount(num)
}

func ptxasParseCount(s string) (int64, bool) {
	if s == "" || strings.Trim(s, "0123456789") != "" {
		return 0, false
	}
	n, err := strconv.ParseInt(s, 10, 64)
	return n, err == nil
}

func (p *ptxasParser) unparsed(raw string) {
	if p.hasCur {
		e := &p.report.Entries[p.cur]
		e.Unparsed = append(e.Unparsed, raw)
		return
	}
	p.report.Unattached = append(p.report.Unattached, raw)
}

func (p *ptxasParser) finish() PtxasReport {
	r := p.report
	r.Counts.Entries = len(r.Entries)
	unparsed := len(r.Unattached)
	for i := range r.Entries {
		e := &r.Entries[i]
		if !e.HasRegisters {
			r.Counts.Incomplete++
		}
		if e.BaseName == "" {
			r.Counts.Undemangled++
		}
		unparsed += len(e.Unparsed)
	}
	r.Counts.UnparsedClauses = unparsed
	r.ArchsSeen = make([]string, 0, len(p.archs))
	for a := range p.archs {
		r.ArchsSeen = append(r.ArchsSeen, a)
	}
	sort.Strings(r.ArchsSeen)
	return r
}

// ptxasDemangleBaseName recovers the namespace-qualified name of a function from its Itanium mangling,
// dropping template arguments and the parameter list. See ptxasDemangle for the accepted subset.
func ptxasDemangleBaseName(mangled string) (string, bool) {
	base, _, ok := ptxasDemangle(mangled)
	return base, ok
}

// ptxasDemangle also returns the template arguments of the final name component, one entry per top-level
// argument: a literal as its value (true, false, or a decimal integer) or ptxasTypeArg for a type or pack,
// so a caller can compare literal arguments position by position. Only the subset ptxas emits for kernels
// is handled: nested and unscoped names, source names, substitutions, literal and type template arguments,
// and a non-empty parameter list. Operators, lambdas, local names, expressions and special names return
// false, so the entry is reported as undemangled instead of guessed. A name without the Itanium prefix is a
// C-linkage symbol whose source name is the symbol.
func ptxasDemangle(mangled string) (string, []string, bool) {
	if !strings.HasPrefix(mangled, "_Z") {
		if !ptxasIsIdentifier(mangled) {
			return "", nil, false
		}
		return mangled, nil, true
	}
	d := ptxasDemangler{s: mangled, pos: 2}
	d.accept('L')
	names, ok := d.encodingName()
	if !ok {
		return "", nil, false
	}
	tail := d.pos
	for d.pos < len(d.s) {
		if !d.skipType() {
			return "", nil, false
		}
	}
	if d.pos == tail {
		return "", nil, false
	}
	return strings.Join(names, "::"), d.leafArgs, true
}

func ptxasIsIdentifier(s string) bool {
	if s == "" || isDigitByte(s[0]) {
		return false
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(isDigitByte(c) || c == '_' || c == '$' || (c|0x20 >= 'a' && c|0x20 <= 'z')) {
			return false
		}
	}
	return true
}

// ptxasDemangler walks one mangled name. subs holds the substitution candidates a name prefix creates, as
// name lists with template arguments stripped; opaque is set once a template argument carries a type,
// because such a type creates candidates this walker does not track, so a later substitution is refused.
type ptxasDemangler struct {
	s        string
	pos      int
	subs     [][]string
	opaque   bool
	leafArgs []string
}

func (d *ptxasDemangler) peek() byte {
	return d.at(0)
}

func (d *ptxasDemangler) at(off int) byte {
	if d.pos+off < 0 || d.pos+off >= len(d.s) {
		return 0
	}
	return d.s[d.pos+off]
}

func (d *ptxasDemangler) accept(c byte) bool {
	if d.peek() != c {
		return false
	}
	d.pos++
	return true
}

func isDigitByte(c byte) bool {
	return c >= '0' && c <= '9'
}

func (d *ptxasDemangler) encodingName() ([]string, bool) {
	switch d.peek() {
	case 'N':
		return d.nestedName(true)
	case 'Z', 0:
		return nil, false
	}
	var names []string
	if strings.HasPrefix(d.s[d.pos:], "St") {
		names = append(names, "std")
		d.pos += 2
	}
	name, ok := d.sourceName()
	if !ok {
		return nil, false
	}
	names = append(names, name)
	if d.peek() == 'I' {
		d.leafArgs = nil
		if !d.templateArgs(&d.leafArgs) {
			return nil, false
		}
	}
	return names, true
}

// nestedName reads N [cv] [ref] component... E. With resolve set, substitutions are looked up and the
// returned names are the qualified name; without it the walk only skips over the name.
func (d *ptxasDemangler) nestedName(resolve bool) ([]string, bool) {
	d.pos++
	for c := d.peek(); c == 'K' || c == 'V' || c == 'r'; c = d.peek() {
		d.pos++
	}
	if !d.accept('R') {
		d.accept('O')
	}
	var names []string
	for {
		c := d.peek()
		switch {
		case c == 'E':
			d.pos++
			return names, len(names) > 0
		case isDigitByte(c):
			name, ok := d.sourceName()
			if !ok {
				return nil, false
			}
			names = append(names, name)
			if !d.afterComponent(names, resolve, true) {
				return nil, false
			}
		case c == 'S' && len(names) > 0:
			// A substitution can only open a nested name; one after a component is outside the grammar.
			return nil, false
		case c == 'S' && d.at(1) == 't':
			d.pos += 2
			names = append(names, "std")
		case c == 'S':
			var ok bool
			names, ok = d.substitutedPrefix(names, resolve)
			if !ok {
				return nil, false
			}
		case (c == 'C' || c == 'D') && isDigitByte(d.at(1)) && len(names) > 0:
			leaf := names[len(names)-1]
			if c == 'D' {
				leaf = "~" + leaf
			}
			names = append(names, leaf)
			d.pos += 2
			if resolve {
				d.leafArgs = nil
			}
		default:
			return nil, false
		}
	}
}

// substitutedPrefix consumes an S-substitution used as a nested-name prefix, which replaces the whole prefix
// built so far, and then any template arguments that follow it.
func (d *ptxasDemangler) substitutedPrefix(names []string, resolve bool) ([]string, bool) {
	idx, ok := d.substitutionIndex()
	if !ok {
		return nil, false
	}
	if resolve {
		if d.opaque || idx >= len(d.subs) {
			return nil, false
		}
		names = append([]string(nil), d.subs[idx]...)
	} else {
		names = append(names, "")
	}
	return names, d.afterComponent(names, resolve, false)
}

// afterComponent records the candidates a nested-name component creates and skips its template arguments.
// A source name yields two candidates when followed by arguments (template prefix, then prefix with
// arguments); a substitution yields only the candidate for its arguments.
func (d *ptxasDemangler) afterComponent(names []string, resolve, addBare bool) bool {
	if resolve && addBare {
		d.subs = append(d.subs, append([]string(nil), names...))
	}
	if resolve {
		d.leafArgs = nil
	}
	if d.peek() != 'I' {
		return true
	}
	if !d.templateArgs(d.leafOut(resolve)) {
		return false
	}
	if resolve {
		d.subs = append(d.subs, append([]string(nil), names...))
	}
	return true
}

// leafOut is where the template arguments of a nested-name component are recorded. Only the top-level walk
// records them; parameter types walked in skip mode must not overwrite the final component's arguments.
func (d *ptxasDemangler) leafOut(resolve bool) *[]string {
	if !resolve {
		return nil
	}
	return &d.leafArgs
}

// substitutionIndex reads S<seq>_ and returns the candidate index it names. The sequence is base 36 with
// digits before capitals, and the bare S_ names candidate zero.
func (d *ptxasDemangler) substitutionIndex() (int, bool) {
	if !d.accept('S') {
		return 0, false
	}
	if d.accept('_') {
		return 0, true
	}
	idx, digits := 0, 0
	for {
		c := d.peek()
		switch {
		case c == '_':
			d.pos++
			return idx + 1, digits > 0
		case isDigitByte(c):
			idx = idx*36 + int(c-'0')
		case c >= 'A' && c <= 'Z':
			idx = idx*36 + int(c-'A') + 10
		default:
			return 0, false
		}
		digits++
		d.pos++
		if digits > ptxasMaxSeqDigits {
			return 0, false
		}
	}
}

// sourceName reads <length><identifier>. A leading zero is refused, as the grammar does not allow it.
func (d *ptxasDemangler) sourceName() (string, bool) {
	start := d.pos
	for isDigitByte(d.peek()) {
		d.pos++
	}
	digits := d.s[start:d.pos]
	if digits == "" || digits[0] == '0' || len(digits) > ptxasMaxSeqDigits {
		return "", false
	}
	n, err := strconv.Atoi(digits)
	if err != nil || d.pos+n > len(d.s) {
		return "", false
	}
	id := d.s[d.pos : d.pos+n]
	d.pos += n
	if !ptxasIsIdentifier(id) {
		return "", false
	}
	if id == ptxasAnonymousNamespace {
		return "(anonymous namespace)", true
	}
	return id, true
}

// skipTemplateArgs consumes I <argument>... E. A literal is candidate-free; any other argument sets opaque
// through skipType.
func (d *ptxasDemangler) skipTemplateArgs() bool {
	return d.templateArgs(nil)
}

// templateArgs consumes I <argument>... E. When out is non-nil, each top-level argument is appended to it:
// a literal as its value, anything else as ptxasTypeArg.
func (d *ptxasDemangler) templateArgs(out *[]string) bool {
	d.pos++
	for d.peek() != 'E' {
		start := d.pos
		switch d.peek() {
		case 0:
			return false
		case 'L':
			if !d.skipLiteral() {
				return false
			}
			if out != nil {
				*out = append(*out, ptxasLiteralValue(d.s[start:d.pos]))
			}
		case 'J':
			d.opaque = true
			d.pos++
			if !d.skipArgPack() {
				return false
			}
			if out != nil {
				*out = append(*out, ptxasTypeArg)
			}
		default:
			if !d.skipType() {
				return false
			}
			if out != nil {
				*out = append(*out, ptxasTypeArg)
			}
		}
	}
	d.pos++
	return true
}

// ptxasLiteralValue renders one L <type code> <value> E literal. A value the parser does not render as a
// plain decimal or bool is returned as ptxasTypeArg, so it can never equal a launch-site literal.
func ptxasLiteralValue(lit string) string {
	if len(lit) < 4 {
		return ptxasTypeArg
	}
	code, value := lit[1], lit[2:len(lit)-1]
	if code == 'b' {
		switch value {
		case "0":
			return "false"
		case "1":
			return "true"
		}
		return ptxasTypeArg
	}
	negative := strings.HasPrefix(value, "n")
	if negative {
		value = value[1:]
	}
	if strings.IndexByte(ptxasIntegerCodes, code) < 0 || value == "" || strings.Trim(value, "0123456789") != "" {
		return ptxasTypeArg
	}
	if negative {
		return "-" + value
	}
	return value
}

// skipArgPack consumes the arguments of a J..E pack, the J already consumed by the caller.
func (d *ptxasDemangler) skipArgPack() bool {
	for d.peek() != 'E' {
		switch d.peek() {
		case 0:
			return false
		case 'L':
			if !d.skipLiteral() {
				return false
			}
		default:
			if !d.skipType() {
				return false
			}
		}
	}
	d.pos++
	return true
}

// skipLiteral consumes L <builtin type> <value> E, for example Li16E or Lb0E.
func (d *ptxasDemangler) skipLiteral() bool {
	d.pos++
	if strings.HasPrefix(d.s[d.pos:], "_Z") || strings.IndexByte(ptxasBuiltinCodes, d.peek()) < 0 || d.peek() == 0 {
		return false
	}
	d.pos++
	start := d.pos
	for d.peek() != 'E' {
		c := d.peek()
		if !(isDigitByte(c) || c == 'n' || c == '_' || (c >= 'a' && c <= 'f')) {
			return false
		}
		d.pos++
	}
	d.pos++
	return d.pos-1 > start
}

// skipType consumes one type production of the subset a kernel signature uses. Builtin types are
// candidate-free; every compound type marks the name opaque.
func (d *ptxasDemangler) skipType() bool {
	c := d.peek()
	switch {
	case c == 0:
		return false
	case strings.IndexByte(ptxasBuiltinCodes, c) >= 0:
		d.pos++
		return true
	case c == 'D':
		return d.skipDType()
	case c == 'P', c == 'R', c == 'O', c == 'K', c == 'V', c == 'r':
		d.opaque = true
		d.pos++
		return d.skipType()
	case isDigitByte(c):
		d.opaque = true
		if _, ok := d.sourceName(); !ok {
			return false
		}
		return d.peek() != 'I' || d.skipTemplateArgs()
	case c == 'N':
		d.opaque = true
		_, ok := d.nestedName(false)
		return ok
	case c == 'S':
		return d.skipSubstitutedType()
	case c == 'T':
		d.opaque = true
		return d.skipTemplateParam()
	case c == 'F':
		return d.skipFunctionType()
	case c == 'A':
		d.opaque = true
		d.pos++
		for isDigitByte(d.peek()) {
			d.pos++
		}
		return d.accept('_') && d.skipType()
	case c == 'M':
		d.opaque = true
		d.pos++
		return d.skipType() && d.skipType()
	}
	return false
}

func (d *ptxasDemangler) skipDType() bool {
	switch d.at(1) {
	case 'n', 'a', 'c', 'd', 'e', 'f', 'h', 'i', 's', 'u':
		d.pos += 2
		return true
	case 'p':
		d.opaque = true
		d.pos += 2
		return d.skipType()
	case 'v':
		d.opaque = true
		d.pos += 2
		for isDigitByte(d.peek()) {
			d.pos++
		}
		return d.accept('_') && d.skipType()
	}
	return false
}

// skipSubstitutedType consumes a substitution used as a type. Resolution is not needed to skip it.
func (d *ptxasDemangler) skipSubstitutedType() bool {
	d.opaque = true
	switch d.at(1) {
	case 't':
		d.pos += 2
		if _, ok := d.sourceName(); !ok {
			return false
		}
	case 'a', 'b', 's', 'i', 'o', 'd':
		d.pos += 2
	default:
		if _, ok := d.substitutionIndex(); !ok {
			return false
		}
	}
	return d.peek() != 'I' || d.skipTemplateArgs()
}

func (d *ptxasDemangler) skipTemplateParam() bool {
	d.pos++
	for isDigitByte(d.peek()) {
		d.pos++
	}
	return d.accept('_') && (d.peek() != 'I' || d.skipTemplateArgs())
}

// skipFunctionType consumes F [Y] return/param types [R|O] E.
func (d *ptxasDemangler) skipFunctionType() bool {
	d.opaque = true
	d.pos++
	d.accept('Y')
	for d.peek() != 'E' {
		if d.peek() == 0 {
			return false
		}
		if (d.peek() == 'R' || d.peek() == 'O') && d.at(1) == 'E' {
			d.pos++
			continue
		}
		if !d.skipType() {
			return false
		}
	}
	d.pos++
	return true
}

// PtxasJoinedSite is one launch site with every ptxas entry that shares its base name. A site that matches
// several instantiations keeps all of them and is flagged Ambiguous, so no instantiation is dropped.
type PtxasJoinedSite struct {
	Site      LaunchSite   `json:"site"`
	BaseName  string       `json:"base_name"`
	Ambiguous bool         `json:"ambiguous"`
	Entries   []PtxasEntry `json:"entries"`
}

// PtxasJoin is the site-to-entry join of one report against one inventory.
type PtxasJoin struct {
	Provenance         string            `json:"provenance"`
	ArchsSeen          []string          `json:"archs_seen"`
	SiteCount          int               `json:"site_count"`
	JoinedSiteCount    int               `json:"joined_site_count"`
	JoinRate           float64           `json:"join_rate"`
	AmbiguousSiteCount int               `json:"ambiguous_site_count"`
	EntryCount         int               `json:"entry_count"`
	JoinedEntryCount   int               `json:"joined_entry_count"`
	Sites              []PtxasJoinedSite `json:"sites"`
	UnjoinedSites      []LaunchSite      `json:"unjoined_sites"`
	UnjoinedEntries    []PtxasEntry      `json:"unjoined_entries"`
}

// PtxasJoinSites joins launch sites to report entries by qualified base name. The inventory name is
// reduced to its part before the first template argument list, so "k<Ratio, true>" joins "k". A site's
// literal template arguments (true, false, integers) must equal the entry's arguments at the same
// position; an entry that disagrees is not joined. Non-literal site arguments (types, expressions) are not
// compared, so a join through them is not verified.
func PtxasJoinSites(report PtxasReport, sites []LaunchSite) PtxasJoin {
	byBase := map[string][]int{}
	for i, e := range report.Entries {
		if e.BaseName != "" {
			byBase[e.BaseName] = append(byBase[e.BaseName], i)
		}
	}
	join := PtxasJoin{
		Provenance: ptxasProvenance + "; arch " + strings.Join(report.ArchsSeen, ","),
		ArchsSeen:  report.ArchsSeen,
		SiteCount:  len(sites),
		EntryCount: len(report.Entries),
	}
	joined := make([]bool, len(report.Entries))
	for _, s := range sites {
		base := ptxasSiteBaseName(s.Kernel)
		idx := ptxasAgreeingEntries(report.Entries, byBase[base], ptxasSiteTemplateArgs(s.Kernel))
		if len(idx) == 0 {
			join.UnjoinedSites = append(join.UnjoinedSites, s)
			continue
		}
		js := PtxasJoinedSite{Site: s, BaseName: base, Ambiguous: len(idx) > 1}
		for _, i := range idx {
			joined[i] = true
			js.Entries = append(js.Entries, report.Entries[i])
		}
		if js.Ambiguous {
			join.AmbiguousSiteCount++
		}
		join.Sites = append(join.Sites, js)
	}
	for i, e := range report.Entries {
		if joined[i] {
			join.JoinedEntryCount++
		} else {
			join.UnjoinedEntries = append(join.UnjoinedEntries, e)
		}
	}
	join.JoinedSiteCount = len(join.Sites)
	if join.SiteCount > 0 {
		join.JoinRate = float64(join.JoinedSiteCount) / float64(join.SiteCount)
	}
	return join
}

func ptxasSiteBaseName(kernel string) string {
	base, _, _ := strings.Cut(kernel, "<")
	return strings.TrimSpace(base)
}

// ptxasSiteTemplateArgs splits the top-level template arguments of an inventory name such as
// "k<Ratio, true>" into their trimmed texts. A name without an argument list has none.
func ptxasSiteTemplateArgs(kernel string) []string {
	open := strings.IndexByte(kernel, '<')
	closing := strings.LastIndexByte(kernel, '>')
	if open < 0 || closing < open {
		return nil
	}
	var args []string
	depth, start := 0, open+1
	for i := open + 1; i < closing; i++ {
		switch kernel[i] {
		case '<', '(':
			depth++
		case '>', ')':
			depth--
		case ',':
			if depth == 0 {
				args = append(args, strings.TrimSpace(kernel[start:i]))
				start = i + 1
			}
		}
	}
	return append(args, strings.TrimSpace(kernel[start:closing]))
}

// ptxasAgreeingEntries keeps the candidate entries whose template arguments match every literal site
// argument at the same position. A site argument that is not a literal is not compared.
func ptxasAgreeingEntries(entries []PtxasEntry, candidates []int, siteArgs []string) []int {
	var out []int
	for _, i := range candidates {
		if ptxasArgsAgree(entries[i].TemplateArgs, siteArgs) {
			out = append(out, i)
		}
	}
	return out
}

func ptxasArgsAgree(entryArgs, siteArgs []string) bool {
	for pos, want := range siteArgs {
		if !ptxasIsLiteralArg(want) {
			continue
		}
		if pos >= len(entryArgs) || entryArgs[pos] != want {
			return false
		}
	}
	return true
}

func ptxasIsLiteralArg(s string) bool {
	if s == "true" || s == "false" {
		return true
	}
	_, err := strconv.ParseInt(s, 10, 64)
	return err == nil
}

// Machine-readable reasons a sweep row is refused. The Refusal text carries the same cause for people.
const (
	PtxasRefusalIncomplete      = "ptxas_incomplete"
	PtxasRefusalSymbolicThreads = "ptxas_symbolic_threads"
	PtxasRefusalLedger          = "ledger_refused"
)

// PtxasSweepRow prices one joined (site, entry) pair on one NVArch row. Units stay in separate fields:
// lane-slot-waves for the lane ledger, registers for the register file, warps and blocks for occupancy.
// A launch that cannot be priced carries its refusal text and zero measurements.
type PtxasSweepRow struct {
	Kernel                      string               `json:"kernel"`
	File                        string               `json:"file"`
	Mangled                     string               `json:"mangled"`
	BaseName                    string               `json:"base_name"`
	Ambiguous                   bool                 `json:"ambiguous"`
	ThreadsPerBlock             int                  `json:"threads_per_block"`
	RegsPerThread               int                  `json:"regs_per_thread"`
	StaticSmemBytes             int64                `json:"static_smem_bytes"`
	SmemDynamicUnknown          bool                 `json:"smem_dynamic_unknown"`
	GridBlocks                  int                  `json:"grid_blocks"`
	Refusal                     string               `json:"refusal,omitempty"`
	RefusalKind                 string               `json:"refusal_kind,omitempty"`
	Limiter                     string               `json:"limiter,omitempty"`
	Waves                       int                  `json:"waves"`
	BlocksPerSM                 int                  `json:"blocks_per_sm"`
	WarpsPerBlock               int                  `json:"warps_per_block"`
	WarpsPerSM                  int                  `json:"warps_per_sm"`
	MaxWarpsPerSM               int                  `json:"max_warps_per_sm"`
	LaneWaveCapacity            int64                `json:"capacity_lane_slot_waves"`
	ActiveLaneSlotWaves         int64                `json:"active_lane_slot_waves"`
	ResourceCappedLaneSlotWaves int64                `json:"resource_capped_lane_slot_waves"`
	LaneFindings                []GapFinding         `json:"lane_findings,omitempty"`
	RegisterFindings            []RegisterGapFinding `json:"register_findings,omitempty"`
	Provenance                  string               `json:"provenance"`
}

// PtxasSweep prices every joined (site, entry) pair on this arch at deviceSMs SMs and gridBlocks blocks.
// The grid is a caller input because the inventory's grid expressions are symbolic. Static shared memory
// comes from ptxas; a site whose dynamic shared-memory expression is not the literal 0 is flagged because
// its footprint is not in the sweep. A log for another architecture is refused whole, not row by row.
func (a NVArch) PtxasSweep(join PtxasJoin, deviceSMs int, gridBlocks int) ([]PtxasSweepRow, error) {
	if deviceSMs <= 0 {
		return nil, ErrPtxasDeviceSMs
	}
	if gridBlocks <= 0 {
		return nil, ErrPtxasGridBlocks
	}
	if len(join.ArchsSeen) == 0 {
		return nil, ErrPtxasNoArch
	}
	for _, seen := range join.ArchsSeen {
		if seen != a.Label {
			return nil, fmt.Errorf("%w: log has %s, NVArch is %s", ErrPtxasArchMismatch, seen, a.Label)
		}
	}
	var rows []PtxasSweepRow
	for _, js := range join.Sites {
		for _, e := range js.Entries {
			rows = append(rows, a.ptxasSweepRow(js, e, deviceSMs, gridBlocks, join.Provenance))
		}
	}
	return rows, nil
}

func (a NVArch) ptxasSweepRow(js PtxasJoinedSite, e PtxasEntry, deviceSMs, gridBlocks int, provenance string) PtxasSweepRow {
	row := PtxasSweepRow{
		Kernel:             js.Site.Kernel,
		File:               js.Site.File,
		Mangled:            e.Mangled,
		BaseName:           e.BaseName,
		Ambiguous:          js.Ambiguous,
		ThreadsPerBlock:    js.Site.ThreadsPerBlock,
		RegsPerThread:      e.Registers,
		StaticSmemBytes:    e.SmemBytes,
		SmemDynamicUnknown: js.Site.SmemExpr != "0",
		GridBlocks:         gridBlocks,
		MaxWarpsPerSM:      a.MaxWarpsPerSM,
		Provenance:         provenance,
	}
	switch {
	case !e.HasRegisters:
		row.Refusal, row.RefusalKind = ErrPtxasIncomplete.Error(), PtxasRefusalIncomplete
		return row
	case js.Site.ThreadsPerBlock <= 0:
		row.Refusal, row.RefusalKind = ErrPtxasSymbolicThreads.Error(), PtxasRefusalSymbolicThreads
		return row
	}
	launch := DecodeLaunch{
		Kernel:            js.Site.Kernel,
		ThreadsPerBlock:   js.Site.ThreadsPerBlock,
		RegsPerThread:     e.Registers,
		SmemBytesPerBlock: int(e.SmemBytes),
		GridBlocks:        gridBlocks,
	}
	ledger, err := a.BuildLaneLedger(launch, deviceSMs)
	if err != nil {
		row.Refusal, row.RefusalKind = err.Error(), PtxasRefusalLedger
		return row
	}
	footprint, err := a.RegisterFootprintOf(launch)
	if err != nil {
		row.Refusal, row.RefusalKind = err.Error(), PtxasRefusalLedger
		return row
	}
	row.Limiter = ledger.Limiter
	row.Waves = ledger.Waves
	row.BlocksPerSM = ledger.BlocksPerSM
	row.WarpsPerBlock = ledger.WarpsPerBlock
	row.WarpsPerSM = ledger.BlocksPerSM * ledger.WarpsPerBlock
	row.LaneWaveCapacity = ledger.Capacity
	row.ActiveLaneSlotWaves = ledger.Cells[CellActive]
	row.ResourceCappedLaneSlotWaves = ledger.Cells[CellResourceCapped]
	row.LaneFindings = ledger.Findings()
	row.RegisterFindings = footprint.Findings(ledger.Limiter)
	return row
}

// RankPtxasSweepRows returns a copy with every priced row before every refused row, then ordered by
// resource-capped lane-slot-waves, largest first, then by kernel and mangled name. A refused row carries
// zero measurements, so it must not tie with a priced zero.
func RankPtxasSweepRows(rows []PtxasSweepRow) []PtxasSweepRow {
	out := append([]PtxasSweepRow(nil), rows...)
	sort.SliceStable(out, func(i, j int) bool {
		if (out[i].Refusal == "") != (out[j].Refusal == "") {
			return out[i].Refusal == ""
		}
		if out[i].ResourceCappedLaneSlotWaves != out[j].ResourceCappedLaneSlotWaves {
			return out[i].ResourceCappedLaneSlotWaves > out[j].ResourceCappedLaneSlotWaves
		}
		if out[i].Kernel != out[j].Kernel {
			return out[i].Kernel < out[j].Kernel
		}
		return out[i].Mangled < out[j].Mangled
	})
	return out
}
