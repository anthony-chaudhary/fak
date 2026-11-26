package compute

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// The launch-site resolver is test-only: it reads the CUDA sources in this package and decides, for each
// launch, whether the thread count is a compile-time integer. It follows declarations through namespaces,
// structs, function scopes and macros, and it refuses whenever the visible definitions disagree or any of
// them is not a constant, so a name it cannot pin down stays symbolic.
//
// Residual: preprocessor conditionals are not interpreted. Every #if, #ifdef and #else arm is replayed, so a
// macro defined only under a build flag reads as active. The inventory's default-valued macros
// (INK2_NV2_WARPS, V41C_THREADS) carry that residual; no build in this tree overrides them.

// launchGenSource is one file of the package: its repo-relative path and its raw text.
type launchGenSource struct {
	file string
	text string
}

type launchGenTokKind uint8

const (
	launchGenIdent launchGenTokKind = iota
	launchGenNumber
	launchGenPunct
	launchGenString
	launchGenDirective
	launchGenOpaque
)

// launchGenTok is one lexeme; off is its byte offset in the blanked text it was read from.
type launchGenTok struct {
	kind launchGenTokKind
	text string
	off  int
}

var (
	launchGenPunct3 = []string{"<<<", ">>>", "<<=", ">>=", "...", "->*"}
	launchGenPunct2 = []string{"::", "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
		"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "##"}
)

func launchGenIsIdentStart(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func launchGenIsIdentByte(c byte) bool {
	return launchGenIsIdentStart(c) || (c >= '0' && c <= '9')
}

func launchGenIsDigit(c byte) bool { return c >= '0' && c <= '9' }

// launchGenTokenize splits blanked source into lexemes. Directives come out as one token per logical line.
func launchGenTokenize(text string) []launchGenTok {
	var out []launchGenTok
	for i := 0; i < len(text); {
		if isSpaceByte(text[i]) {
			i++
			continue
		}
		kind, end := launchGenScanToken(text, i)
		out = append(out, launchGenTok{kind: kind, text: text[i:end], off: i})
		i = end
	}
	return out
}

func launchGenScanToken(text string, i int) (launchGenTokKind, int) {
	c := text[i]
	switch {
	case c == '#' && launchGenAtLineStart(text, i):
		if j := strings.IndexByte(text[i:], '\n'); j >= 0 {
			return launchGenDirective, i + j
		}
		return launchGenDirective, len(text)
	case launchGenIsIdentStart(c):
		j := i + 1
		for j < len(text) && launchGenIsIdentByte(text[j]) {
			j++
		}
		return launchGenIdent, j
	case launchGenIsDigit(c) || (c == '.' && i+1 < len(text) && launchGenIsDigit(text[i+1])):
		return launchGenNumber, launchGenNumberEnd(text, i)
	case c == '"' || c == '\'':
		if j := strings.IndexByte(text[i+1:], c); j >= 0 {
			return launchGenString, i + j + 2
		}
		return launchGenString, len(text)
	}
	for _, p := range launchGenPunct3 {
		if strings.HasPrefix(text[i:], p) {
			return launchGenPunct, i + 3
		}
	}
	for _, p := range launchGenPunct2 {
		if strings.HasPrefix(text[i:], p) {
			return launchGenPunct, i + 2
		}
	}
	return launchGenPunct, i + 1
}

func launchGenAtLineStart(text string, i int) bool {
	for j := i - 1; j >= 0; j-- {
		switch text[j] {
		case '\n':
			return true
		case ' ', '\t':
			continue
		default:
			return false
		}
	}
	return true
}

// launchGenNumberEnd reads a numeric literal, including a signed exponent outside hexadecimal.
func launchGenNumberEnd(text string, i int) int {
	hex := i+1 < len(text) && text[i] == '0' && (text[i+1] == 'x' || text[i+1] == 'X')
	j := i + 1
	for j < len(text) {
		c := text[j]
		switch {
		case launchGenIsIdentByte(c) || c == '.':
		case (c == '+' || c == '-') && !hex && (text[j-1] == 'e' || text[j-1] == 'E'):
		default:
			return j
		}
		j++
	}
	return j
}

// launchGenIntLiteral parses a decimal, hexadecimal or octal integer literal with optional suffixes.
func launchGenIntLiteral(text string) (int64, error) {
	s := strings.TrimRight(text, "uUlLzZ")
	if s == "" || launchGenHasFloatMark(s) {
		return 0, fmt.Errorf("%w: %q is not an integer literal", launchGenErrUnresolved, text)
	}
	var n int64
	base := int64(10)
	switch {
	case strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X"):
		base, s = 16, s[2:]
	case len(s) > 1 && s[0] == '0':
		base, s = 8, s[1:]
	}
	for _, c := range s {
		d, ok := launchGenDigitValue(c)
		if !ok || d >= base {
			return 0, fmt.Errorf("%w: %q is not an integer literal", launchGenErrUnresolved, text)
		}
		n = n*base + d
	}
	return n, nil
}

func launchGenHasFloatMark(s string) bool {
	return strings.ContainsAny(s, ".pP") || (!strings.HasPrefix(s, "0x") && strings.ContainsAny(s, "eE"))
}

func launchGenDigitValue(c rune) (int64, bool) {
	switch {
	case c >= '0' && c <= '9':
		return int64(c - '0'), true
	case c >= 'a' && c <= 'f':
		return int64(c-'a') + 10, true
	case c >= 'A' && c <= 'F':
		return int64(c-'A') + 10, true
	}
	return 0, false
}

// launchGenMacro is one #define or #undef of an object-like or function-like macro.
type launchGenMacro struct {
	name   string
	off    int
	undef  bool
	fnLike bool
	body   []launchGenTok
}

func launchGenDirectiveMacro(text string, off int) (launchGenMacro, bool) {
	rest := strings.TrimLeft(strings.TrimPrefix(text, "#"), " \t")
	word, rest := launchGenLeadIdent(rest)
	name, after := launchGenLeadIdent(strings.TrimLeft(rest, " \t"))
	if name == "" {
		return launchGenMacro{}, false
	}
	switch word {
	case "define":
		m := launchGenMacro{name: name, off: off, fnLike: strings.HasPrefix(after, "(")}
		if !m.fnLike {
			m.body = launchGenTokenize(after)
		}
		return m, true
	case "undef":
		return launchGenMacro{name: name, off: off, undef: true}, true
	}
	return launchGenMacro{}, false
}

func launchGenLeadIdent(s string) (string, string) {
	j := 0
	for j < len(s) && launchGenIsIdentByte(s[j]) {
		j++
	}
	return s[:j], s[j:]
}

type launchGenFrameKind uint8

const (
	launchGenFrameNS launchGenFrameKind = iota
	launchGenFrameStruct
	launchGenFrameFunc
)

// launchGenFrame is one lexical scope that holds declarations. Namespace frames hold nothing themselves:
// their declarations live in launchGenIndex.ns under the namespace path.
type launchGenFrame struct {
	kind     launchGenFrameKind
	parent   *launchGenFrame
	ns       []string
	decls    map[string][]*launchGenDecl
	assigned map[string]bool
}

func launchGenNewFrame(kind launchGenFrameKind, parent *launchGenFrame, ns []string) *launchGenFrame {
	return &launchGenFrame{kind: kind, parent: parent, ns: ns,
		decls: map[string][]*launchGenDecl{}, assigned: map[string]bool{}}
}

type launchGenDeclKind uint8

const (
	launchGenDeclUnknown  launchGenDeclKind = iota // parameter, uninitialised or run-time value
	launchGenDeclTemplate                          // template parameter: unknown, and makes the result template-dependent
	launchGenDeclConst                             // const or constexpr integer or dim3 with an initializer
	launchGenDeclMutable                           // non-const variable, a constant only if never assigned in its function
	launchGenDeclEnum                              // enumerator: explicit initializer or predecessor plus one
	launchGenDeclFunc                              // function definition with a body
	launchGenDeclPreset                            // parameter bound to an evaluated argument during a call
)

type launchGenDecl struct {
	name     string
	kind     launchGenDeclKind
	dim      bool
	ctorArgs bool
	file     string
	off      int
	frame    *launchGenFrame
	init     []launchGenTok
	prev     *launchGenDecl
	params   []string
	inner    *launchGenFrame
	body     []launchGenTok
	val      launchGenVal
}

// launchGenVal is an evaluated thread expression: a plain integer or a dim3 with three components.
type launchGenVal struct {
	n   int64
	dim bool
	d   [3]int64
}

// threads is the thread count of a launch: the value itself, or the product of a dim3.
func (v launchGenVal) threads() int64 {
	if !v.dim {
		return v.n
	}
	return v.d[0] * v.d[1] * v.d[2]
}

// launchGenLaunch is one triple-angle launch with the scope it appears in. viaMacro marks a launch written in a
// macro body, whose names resolve through macros only.
type launchGenLaunch struct {
	file     string
	off      int
	frame    *launchGenFrame
	scan     scannedLaunch
	viaMacro bool
}

// launchGenIndex holds every declaration, macro and launch of the sources, keyed for lookup.
type launchGenIndex struct {
	ns        map[string]map[string][]*launchGenDecl
	classes   map[string]*launchGenFrame
	macros    map[string][]launchGenMacro
	usingNS   map[string][][]string
	usingDecl map[string]map[string][]string
	launches  []launchGenLaunch
}

func launchGenNewIndex() *launchGenIndex {
	return &launchGenIndex{
		ns:        map[string]map[string][]*launchGenDecl{},
		classes:   map[string]*launchGenFrame{},
		macros:    map[string][]launchGenMacro{},
		usingNS:   map[string][][]string{},
		usingDecl: map[string]map[string][]string{},
	}
}

func (idx *launchGenIndex) addNS(ns []string, d *launchGenDecl) {
	key := strings.Join(ns, "::")
	if idx.ns[key] == nil {
		idx.ns[key] = map[string][]*launchGenDecl{}
	}
	idx.ns[key][d.name] = append(idx.ns[key][d.name], d)
}

// lookup returns the declarations a name denotes at ctx: the innermost scope that declares it hides the
// rest, as in C++. A using-directive or using-declaration in the same file adds its namespace after the
// lexical scopes.
func (idx *launchGenIndex) lookup(ctx launchGenCtx, qual []string, name string) []*launchGenDecl {
	ns := ctx.frame.ns
	if len(qual) > 0 {
		return idx.nsLookup(ns, qual, name)
	}
	for f := ctx.frame; f != nil && f.kind != launchGenFrameNS; f = f.parent {
		if ds := f.decls[name]; len(ds) > 0 {
			return ds
		}
	}
	if ds := idx.nsLookup(ns, nil, name); len(ds) > 0 {
		return ds
	}
	for _, u := range idx.usingNS[ctx.file] {
		if ds := idx.nsLookup(nil, u, name); len(ds) > 0 {
			return ds
		}
	}
	if pre, ok := idx.usingDecl[ctx.file][name]; ok {
		return idx.nsLookup(ns, pre, name)
	}
	return nil
}

// nsLookup searches qual relative to each enclosing namespace of ns, innermost first, then globally.
func (idx *launchGenIndex) nsLookup(ns, qual []string, name string) []*launchGenDecl {
	for k := len(ns); k >= 0; k-- {
		path := append(append([]string(nil), ns[:k]...), qual...)
		if ds := idx.ns[strings.Join(path, "::")][name]; len(ds) > 0 {
			return ds
		}
	}
	return nil
}

// macroEvents counts the define and undef directives of name in file. A macro body is expanded where it is
// used, so a name with more than one event cannot be placed at the expansion site.
func (idx *launchGenIndex) macroEvents(file, name string) int {
	n := 0
	for _, m := range idx.macros[file] {
		if m.name == name {
			n++
		}
	}
	return n
}

// macrosVisible returns the object-like or function-like definitions of name that reach offset off of file.
// Positional state within the file wins; a name not defined there at that point falls back to the final
// state of every other file, since headers are not tracked per translation unit.
func (idx *launchGenIndex) macrosVisible(file string, off int, name string) []launchGenMacro {
	if state := launchGenMacroState(idx.macros[file], name, off); len(state) > 0 {
		return state
	}
	files := make([]string, 0, len(idx.macros))
	for f := range idx.macros {
		if f != file {
			files = append(files, f)
		}
	}
	sort.Strings(files)
	var out []launchGenMacro
	for _, f := range files {
		out = append(out, launchGenMacroState(idx.macros[f], name, -1)...)
	}
	return out
}

// launchGenMacroState replays the directives of one file for name. A negative before replays all of them.
func launchGenMacroState(evs []launchGenMacro, name string, before int) []launchGenMacro {
	var state []launchGenMacro
	for _, m := range evs {
		if m.name != name || (before >= 0 && m.off >= before) {
			continue
		}
		if m.undef {
			state = nil
		} else {
			state = append(state, m)
		}
	}
	return state
}

// launchGenCtx is a point in the sources: the file, the offset, and the innermost scope there. viaMacro is
// set for a macro body, which is expanded at each use, so only macro names are read from it.
type launchGenCtx struct {
	file     string
	off      int
	frame    *launchGenFrame
	viaMacro bool
}

// launchGenWalker turns one file's tokens into scope frames, declarations and launches.
type launchGenWalker struct {
	idx   *launchGenIndex
	file  string
	text  string
	toks  []launchGenTok
	cur   *launchGenFrame
	stack []launchGenScope
	buf   []launchGenTok
	depth int
}

// launchGenScope is an open brace. A function scope sets fn so its body is recorded at the closing brace.
// An init scope is a brace inside an expression (a lambda or a braced initializer): the statement it sits
// in continues after the brace, so the saved buffer is restored on close.
type launchGenScope struct {
	prev       *launchGenFrame
	fn         *launchGenDecl
	start      int
	init       bool
	savedBuf   []launchGenTok
	savedDepth int
}

func (w *launchGenWalker) run() error {
	for i := 0; i < len(w.toks); {
		t := w.toks[i]
		var err error
		switch {
		case t.kind == launchGenDirective:
			if m, ok := launchGenDirectiveMacro(t.text, t.off); ok {
				w.idx.macros[w.file] = append(w.idx.macros[w.file], m)
			}
			if err = w.directiveLaunches(t); err != nil {
				return err
			}
			i++
		case t.kind != launchGenPunct:
			w.buf = append(w.buf, t)
			i++
		case t.text == "<<<":
			i, err = w.launch(i)
		case t.text == "{" && w.depth > 0:
			w.pushLambdaOrInit(w.buf, i)
			i++
		case t.text == "{":
			i, err = w.open(i)
		case t.text == "}":
			i, err = w.close(i)
		case t.text == ";" && w.depth == 0:
			w.statement(w.buf)
			w.buf = nil
			i++
		case t.text == "(":
			w.depth++
			w.buf = append(w.buf, t)
			i++
		case t.text == ")":
			w.depth--
			w.buf = append(w.buf, t)
			i++
		case t.text == ":" && w.depth == 0 && launchGenIsLabel(w.buf):
			w.buf = nil
			i++
		default:
			w.buf = append(w.buf, t)
			i++
		}
		if err != nil {
			return err
		}
	}
	if len(w.stack) > 0 {
		return errors.New("unclosed brace at end of file")
	}
	return nil
}

func launchGenIsLabel(buf []launchGenTok) bool {
	if len(buf) == 0 {
		return false
	}
	switch buf[0].text {
	case "case", "default", "public", "private", "protected":
		return true
	}
	return false
}

// directiveLaunches records the launches written in a directive, which are macro bodies in practice. The
// text scanner counts them too, so the walker must find the same sites.
func (w *launchGenWalker) directiveLaunches(t launchGenTok) error {
	for r := strings.Index(t.text, "<<<"); r >= 0; {
		off := t.off + r
		kernel, err := kernelBefore(w.text, off)
		if err != nil {
			return fmt.Errorf("%s: %w", w.file, err)
		}
		args, _, err := launchConfigArgs(w.text, off+3)
		if err != nil {
			return fmt.Errorf("%s: %w", w.file, err)
		}
		scan := scannedLaunch{Kernel: kernel, GridExpr: args[0], ThreadsExpr: args[1]}
		if len(args) > 2 {
			scan.SmemExpr = args[2]
		}
		w.idx.launches = append(w.idx.launches, launchGenLaunch{file: w.file, off: off, frame: w.cur, scan: scan, viaMacro: true})
		next := strings.Index(t.text[r+3:], "<<<")
		if next < 0 {
			break
		}
		r += 3 + next
	}
	return nil
}

// launch records a triple-angle launch with the configuration read from the blanked text, and replaces its
// tokens with one opaque token so the enclosing statement still parses.
func (w *launchGenWalker) launch(i int) (int, error) {
	off := w.toks[i].off
	kernel, err := kernelBefore(w.text, off)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", w.file, err)
	}
	args, end, err := launchConfigArgs(w.text, off+3)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", w.file, err)
	}
	scan := scannedLaunch{Kernel: kernel, GridExpr: args[0], ThreadsExpr: args[1]}
	if len(args) > 2 {
		scan.SmemExpr = args[2]
	}
	w.idx.launches = append(w.idx.launches, launchGenLaunch{file: w.file, off: off, frame: w.cur, scan: scan})
	w.buf = append(w.buf, launchGenTok{kind: launchGenOpaque, text: "<<<>>>", off: off})
	j := i + 1
	for j < len(w.toks) && w.toks[j].off < end {
		j++
	}
	return j, nil
}

// open classifies the brace at token i by the header tokens in the buffer.
func (w *launchGenWalker) open(i int) (int, error) {
	header := w.buf
	w.buf = nil
	rest, tparams := launchGenStripTemplate(header)
	first := ""
	if len(rest) > 0 {
		first = rest[0].text
	}
	switch {
	case launchGenHasTopAssign(rest):
		w.pushLambdaOrInit(header, i)
		return i + 1, nil
	case first == "enum":
		end, err := launchGenMatch(w.toks, i, "{", "}")
		if err != nil {
			return 0, fmt.Errorf("%s: %w", w.file, err)
		}
		w.enumerators(w.toks[i+1 : end])
		return end + 1, nil
	case first == "namespace":
		if path := launchGenQualPath(launchGenDropWord(rest[1:], "inline")); len(path) > 0 {
			w.push(launchGenNewFrame(launchGenFrameNS, w.cur, append(append([]string(nil), w.cur.ns...), path...)), i)
		} else {
			w.push(w.cur, i)
		}
	case first == "struct" || first == "class" || first == "union":
		frame := launchGenNewFrame(launchGenFrameStruct, w.cur, w.cur.ns)
		w.declareTemplateParams(frame, tparams)
		if len(rest) > 1 && rest[1].kind == launchGenIdent {
			w.idx.classes[strings.Join(append(append([]string(nil), w.cur.ns...), rest[1].text), "::")] = frame
		}
		w.push(frame, i)
	case first == "for":
		w.forHeader(rest)
		w.push(w.cur, i)
	case launchGenIsControl(first) || first == "" || launchGenIsExternBlock(rest):
		w.markAssigned(rest)
		w.push(w.cur, i)
	default:
		if sig, ok := launchGenSigOf(rest); ok && !sig.lambda {
			w.pushFunc(sig, tparams, i, false, header)
			return i + 1, nil
		}
		w.markAssigned(rest)
		w.push(w.cur, i)
	}
	return i + 1, nil
}

// launchGenIsExternBlock matches the linkage block extern "C" {, not a function with C linkage.
func launchGenIsExternBlock(rest []launchGenTok) bool {
	return len(rest) == 2 && rest[0].text == "extern" && rest[1].kind == launchGenString
}

func (w *launchGenWalker) push(frame *launchGenFrame, i int) {
	w.stack = append(w.stack, launchGenScope{prev: w.cur, start: i + 1})
	w.cur = frame
}

// pushLambdaOrInit opens a brace inside an expression. A lambda gets its own function frame so its
// parameters shadow outer names; any other brace is an init scope.
func (w *launchGenWalker) pushLambdaOrInit(header []launchGenTok, i int) {
	saved := w.depth
	w.depth = 0
	w.buf = nil
	if sig, ok := launchGenSigOf(header); ok && sig.lambda {
		w.pushFunc(sig, nil, i, true, header)
		w.stack[len(w.stack)-1].savedDepth = saved
		return
	}
	w.stack = append(w.stack, launchGenScope{prev: w.cur, init: true, start: i + 1, savedBuf: header, savedDepth: saved})
}

// pushFunc opens a function body: its parameters and template parameters become a frame of their own, and the
// function is registered under its name in the enclosing scope so calls can be evaluated.
func (w *launchGenWalker) pushFunc(sig launchGenSig, tparams []string, i int, init bool, header []launchGenTok) {
	// An out-of-class definition S::f sees the members of S before the enclosing namespaces. A class defined
	// in another file is not indexed, so its members stay invisible and the lookup falls back to the namespaces.
	ns := append(append([]string(nil), w.cur.ns...), sig.qual...)
	parent := w.cur
	if class := w.idx.classes[strings.Join(ns, "::")]; len(sig.qual) > 0 && class != nil {
		parent = class
	}
	inner := launchGenNewFrame(launchGenFrameFunc, parent, ns)
	w.declareTemplateParams(inner, tparams)
	fd := &launchGenDecl{name: sig.name, kind: launchGenDeclFunc, file: w.file, off: sig.nameOff, frame: w.cur, inner: inner}
	for _, piece := range launchGenSplitTop(sig.params, ",") {
		if len(piece) == 1 && piece[0].text == "void" {
			break
		}
		pname := launchGenLastIdent(piece)
		fd.params = append(fd.params, pname)
		if pname != "" {
			inner.decls[pname] = append(inner.decls[pname], &launchGenDecl{name: pname, kind: launchGenDeclUnknown,
				file: w.file, off: piece[len(piece)-1].off, frame: inner})
		}
	}
	if sig.name != "" && !init {
		w.register(fd)
	}
	s := launchGenScope{prev: w.cur, start: i + 1, init: init, savedBuf: header, savedDepth: w.depth}
	if !init {
		s.fn = fd
	}
	w.stack = append(w.stack, s)
	w.cur = inner
}

func (w *launchGenWalker) declareTemplateParams(frame *launchGenFrame, names []string) {
	for _, n := range names {
		frame.decls[n] = append(frame.decls[n], &launchGenDecl{name: n, kind: launchGenDeclTemplate, frame: frame})
	}
}

// close pops the innermost brace, recording a function body or restoring the statement of an init scope.
func (w *launchGenWalker) close(i int) (int, error) {
	n := len(w.stack)
	if n == 0 {
		return 0, fmt.Errorf("%s: unbalanced brace at offset %d", w.file, w.toks[i].off)
	}
	s := w.stack[n-1]
	w.stack = w.stack[:n-1]
	if s.fn != nil {
		s.fn.body = w.toks[s.start:i]
	}
	w.cur = s.prev
	if s.init {
		w.buf = append(append([]launchGenTok(nil), s.savedBuf...), launchGenTok{kind: launchGenOpaque, text: "{}", off: w.toks[i].off})
		w.depth = s.savedDepth
	} else {
		w.buf = nil
		w.depth = 0
	}
	return i + 1, nil
}

// statement reads one declaration or expression statement at the current scope.
func (w *launchGenWalker) statement(buf []launchGenTok) {
	if len(buf) == 0 {
		return
	}
	w.markReferenceAliases(buf)
	switch buf[0].text {
	case "using":
		w.using(buf)
		return
	case "for":
		w.forHeader(buf)
		return
	case "typedef", "template", "static_assert", "friend", "return", "namespace", "public", "private",
		"protected", "operator", "goto", "throw", "delete", "break", "continue", "case", "default", "asm",
		"if", "while", "do", "switch", "else", "co_return", "new":
		w.markAssigned(buf)
		return
	}
	if !w.declare(buf) {
		w.markAssigned(buf)
	}
}

// declare registers the declarators of a declaration. It returns false when the statement is not one.
func (w *launchGenWalker) declare(buf []launchGenTok) bool {
	pieces, balanced := launchGenSplitDecl(buf)
	if !balanced {
		w.registerUnreadDeclarators(buf)
	}
	head := pieces[0]
	stop := launchGenDeclStop(head)
	if stop < 2 || head[stop-1].kind != launchGenIdent {
		return false
	}
	base := head[:stop-1]
	if !launchGenIsTypeSpec(base) {
		return false
	}
	if stop < len(head) && head[stop].text == "(" && !launchGenHasWord(base, "dim3") {
		return true
	}
	kind, dim := launchGenKindOf(base, false)
	w.declareOne(head[stop-1], kind, dim, head[stop:])
	for _, piece := range pieces[1:] {
		if !balanced {
			break
		}
		j, ptr := 0, false
		for j < len(piece) && piece[j].kind == launchGenPunct && (piece[j].text == "*" || piece[j].text == "&") {
			ptr = true
			j++
		}
		if j < len(piece) && piece[j].kind == launchGenIdent {
			kind, dim := launchGenKindOf(base, ptr)
			w.declareOne(piece[j], kind, dim, piece[j+1:])
		}
	}
	return true
}

// registerUnreadDeclarators records, as unknown, the first name of each comma-separated piece after the first. It is
// used when the angle brackets of a statement do not close, so a comma may be an operator and the pieces cannot
// be read; a shadowing name then never resolves to an outer constant.
func (w *launchGenWalker) registerUnreadDeclarators(buf []launchGenTok) {
	for _, piece := range launchGenSplitTop(buf, ",")[1:] {
		j := 0
		for j < len(piece) && piece[j].kind == launchGenPunct && (piece[j].text == "*" || piece[j].text == "&") {
			j++
		}
		if j < len(piece) && piece[j].kind == launchGenIdent {
			w.register(&launchGenDecl{name: piece[j].text, kind: launchGenDeclUnknown, file: w.file, off: piece[j].off, frame: w.cur})
		}
	}
}

// launchGenSplitDecl splits a declaration at its top-level commas, counting angle brackets as nesting so the
// arguments of a template-id stay with their declarator. balanced is false when the brackets do not close.
func launchGenSplitDecl(toks []launchGenTok) (pieces [][]launchGenTok, balanced bool) {
	if len(toks) == 0 {
		return [][]launchGenTok{nil}, true
	}
	depth, start := 0, 0
	balanced = true
	for j, t := range toks {
		if t.kind != launchGenPunct {
			continue
		}
		switch t.text {
		case "(", "[", "{", "<":
			depth++
		case ")", "]", "}", ">":
			depth--
		case ">>":
			depth -= 2
		case ",":
			if depth == 0 {
				pieces = append(pieces, toks[start:j])
				start = j + 1
			}
		}
		if depth < 0 {
			depth, balanced = 0, false
		}
	}
	if depth != 0 {
		balanced = false
	}
	return append(pieces, toks[start:]), balanced
}

func (w *launchGenWalker) declareOne(name launchGenTok, kind launchGenDeclKind, dim bool, rem []launchGenTok) {
	init, ctor := launchGenInitOf(rem)
	if ctor && !dim {
		kind, init = launchGenDeclUnknown, nil
	}
	w.register(&launchGenDecl{name: name.text, kind: kind, dim: dim, ctorArgs: ctor, file: w.file, off: name.off,
		frame: w.cur, init: append([]launchGenTok(nil), init...)})
	w.markAssigned(init)
}

// register adds a declaration to the scope it belongs to: the namespace table at namespace scope, otherwise
// the frame's own table.
func (w *launchGenWalker) register(d *launchGenDecl) {
	if d.frame == nil {
		d.frame = w.cur
	}
	if w.cur.kind == launchGenFrameNS {
		w.idx.addNS(w.cur.ns, d)
		return
	}
	w.cur.decls[d.name] = append(w.cur.decls[d.name], d)
}

// markAssigned records every name the tokens write to or take the address of, in every enclosing function.
func (w *launchGenWalker) markAssigned(toks []launchGenTok) {
	for j, t := range toks {
		switch {
		case t.kind == launchGenIdent && j+1 < len(toks) && toks[j+1].kind == launchGenPunct && launchGenIsAssignOp(toks[j+1].text):
			w.markName(t.text)
		case t.kind == launchGenIdent && launchGenIsMemberWrite(toks, j):
			// A write to a member of a dim3 changes the object, so the object is no longer its declared value.
			w.markName(t.text)
		case t.kind == launchGenPunct && (t.text == "++" || t.text == "--"):
			if j+1 < len(toks) && toks[j+1].kind == launchGenIdent {
				w.markName(toks[j+1].text)
			}
			if j > 0 && toks[j-1].kind == launchGenIdent {
				w.markName(toks[j-1].text)
			}
			if j >= 3 && toks[j-2].text == "." && toks[j-3].kind == launchGenIdent {
				w.markName(toks[j-3].text)
			}
		case t.kind == launchGenPunct && t.text == "&" && j+1 < len(toks) && toks[j+1].kind == launchGenIdent:
			w.markName(toks[j+1].text)
		}
	}
}

// launchGenIsMemberWrite reports whether toks[j] is the object of a member write: obj . member followed by an
// assignment operator or an increment or decrement.
func launchGenIsMemberWrite(toks []launchGenTok, j int) bool {
	if j+3 >= len(toks) || toks[j+1].text != "." || toks[j+2].kind != launchGenIdent {
		return false
	}
	op := toks[j+3]
	return op.kind == launchGenPunct && (launchGenIsAssignOp(op.text) || op.text == "++" || op.text == "--")
}

// markReferenceAliases marks the referred name as written when a reference is bound to it, because a write
// through the reference is not tracked back to that name.
func (w *launchGenWalker) markReferenceAliases(toks []launchGenTok) {
	for j := 0; j+3 < len(toks); j++ {
		if toks[j].kind == launchGenPunct && toks[j].text == "&" && toks[j+1].kind == launchGenIdent &&
			toks[j+2].kind == launchGenPunct && toks[j+2].text == "=" && toks[j+3].kind == launchGenIdent {
			w.markName(toks[j+3].text)
		}
	}
}

func (w *launchGenWalker) markName(name string) {
	for f := w.cur; f != nil; f = f.parent {
		if f.kind == launchGenFrameFunc {
			f.assigned[name] = true
		}
	}
}

// forHeader registers the declaration of a for-loop's first clause and marks the names the later clauses
// write to. A range-for variable is registered as unknown.
func (w *launchGenWalker) forHeader(toks []launchGenTok) {
	closeAt, err := launchGenMatch(toks, 1, "(", ")")
	if err != nil || len(toks) < 2 || toks[1].text != "(" {
		w.markAssigned(toks)
		return
	}
	for k, c := range launchGenSplitTop(toks[2:closeAt], ";") {
		if k > 0 {
			w.markAssigned(c)
			continue
		}
		if j := launchGenTopIndex(c, ":"); j > 0 && c[j-1].kind == launchGenIdent {
			w.register(&launchGenDecl{name: c[j-1].text, kind: launchGenDeclUnknown, file: w.file, off: c[j-1].off, frame: w.cur})
			continue
		}
		w.statement(c)
	}
	w.markAssigned(toks[closeAt+1:])
}

func (w *launchGenWalker) enumerators(toks []launchGenTok) {
	var prev *launchGenDecl
	for _, piece := range launchGenSplitTop(toks, ",") {
		if len(piece) == 0 || piece[0].kind != launchGenIdent {
			continue
		}
		d := &launchGenDecl{name: piece[0].text, kind: launchGenDeclEnum, file: w.file, off: piece[0].off, frame: w.cur, prev: prev}
		if len(piece) > 2 && piece[1].text == "=" {
			d.init = append([]launchGenTok(nil), piece[2:]...)
		}
		prev = d
		w.register(d)
	}
}

// using records a using-directive or a using-declaration; alias declarations carry no value and are skipped.
func (w *launchGenWalker) using(buf []launchGenTok) {
	rest := buf[1:]
	if len(rest) > 0 && rest[0].text == "namespace" {
		if path := launchGenQualPath(rest[1:]); len(path) > 0 {
			w.idx.usingNS[w.file] = append(w.idx.usingNS[w.file], path)
		}
		return
	}
	if launchGenHasTopAssign(rest) {
		return
	}
	path := launchGenQualPath(rest)
	if len(path) < 2 {
		return
	}
	if w.idx.usingDecl[w.file] == nil {
		w.idx.usingDecl[w.file] = map[string][]string{}
	}
	w.idx.usingDecl[w.file][path[len(path)-1]] = path[:len(path)-1]
}

// launchGenSigOf reads a function header: the last top-level parenthesis group must follow an identifier (the
// function name) or a lambda capture list. Any tokens after that group are qualifiers or a trailing return type.
func launchGenSigOf(toks []launchGenTok) (launchGenSig, bool) {
	open, close, ok := launchGenLastGroup(toks)
	if !ok || open == 0 {
		return launchGenSig{}, false
	}
	prev := toks[open-1]
	switch {
	case prev.text == "]" && prev.kind == launchGenPunct:
		return launchGenSig{lambda: true, params: toks[open+1 : close]}, true
	case prev.kind == launchGenIdent && !launchGenIsControl(prev.text):
		return launchGenSig{name: prev.text, nameOff: prev.off, params: toks[open+1 : close],
			qual: launchGenTrailingQual(toks[:open-1])}, true
	}
	return launchGenSig{}, false
}

// launchGenTrailingQual returns the qualifier path written in front of a declarator name, as in S::f.
func launchGenTrailingQual(toks []launchGenTok) []string {
	start := len(toks)
	for start >= 2 && toks[start-1].text == "::" && toks[start-2].kind == launchGenIdent {
		start -= 2
	}
	return launchGenQualPath(toks[start:])
}

type launchGenSig struct {
	name    string
	nameOff int
	params  []launchGenTok
	lambda  bool
	qual    []string
}

// launchGenLastGroup returns the indices of the last top-level parenthesis group, which closes at close.
func launchGenLastGroup(toks []launchGenTok) (open, close int, ok bool) {
	depth, start := 0, -1
	for j, t := range toks {
		if t.kind != launchGenPunct {
			continue
		}
		switch t.text {
		case "(":
			if depth == 0 {
				start = j
			}
			depth++
		case ")":
			if depth == 0 {
				continue
			}
			depth--
			if depth == 0 {
				open, close, ok = start, j, true
			}
		}
	}
	return open, close, ok
}

// launchGenStripTemplate removes a leading template<...> and returns its parameter names.
func launchGenStripTemplate(toks []launchGenTok) ([]launchGenTok, []string) {
	if len(toks) < 2 || toks[0].text != "template" || toks[1].text != "<" {
		return toks, nil
	}
	depth, end := 0, -1
	for j := 1; j < len(toks); j++ {
		switch toks[j].text {
		case "<":
			depth++
		case ">":
			depth--
		case ">>":
			depth -= 2
		}
		if depth <= 0 {
			end = j
			break
		}
	}
	if end < 0 {
		return toks, nil
	}
	var names []string
	for _, piece := range launchGenSplitTop(toks[2:end], ",") {
		if eq := launchGenTopIndex(piece, "="); eq >= 0 {
			piece = piece[:eq]
		}
		if n := launchGenLastIdent(piece); n != "" {
			names = append(names, n)
		}
	}
	return toks[end+1:], names
}

// launchGenSplitTop splits toks at each top-level separator, where parentheses, brackets and braces nest.
func launchGenSplitTop(toks []launchGenTok, sep string) [][]launchGenTok {
	if len(toks) == 0 {
		return nil
	}
	var out [][]launchGenTok
	depth, start := 0, 0
	for j, t := range toks {
		if t.kind == launchGenPunct {
			switch t.text {
			case "(", "[", "{":
				depth++
			case ")", "]", "}":
				depth--
			}
			if depth == 0 && t.text == sep {
				out = append(out, toks[start:j])
				start = j + 1
			}
		}
	}
	return append(out, toks[start:])
}

func launchGenTopIndex(toks []launchGenTok, sep string) int {
	depth := 0
	for j, t := range toks {
		if t.kind != launchGenPunct {
			continue
		}
		switch t.text {
		case "(", "[", "{":
			depth++
		case ")", "]", "}":
			depth--
		case sep:
			if depth == 0 {
				return j
			}
		}
	}
	return -1
}

func launchGenHasTopAssign(toks []launchGenTok) bool {
	return launchGenTopIndex(toks, "=") >= 0
}

// launchGenMatch returns the index of the closer that matches the opener at open.
func launchGenMatch(toks []launchGenTok, open int, opener, closer string) (int, error) {
	depth := 0
	for j := open; j < len(toks); j++ {
		if toks[j].kind != launchGenPunct {
			continue
		}
		switch toks[j].text {
		case opener:
			depth++
		case closer:
			depth--
			if depth == 0 {
				return j, nil
			}
		}
	}
	return 0, fmt.Errorf("%w: unmatched %s", launchGenErrUnresolved, opener)
}

func launchGenLastIdent(toks []launchGenTok) string {
	for j := len(toks) - 1; j >= 0; j-- {
		if toks[j].kind == launchGenIdent {
			return toks[j].text
		}
	}
	return ""
}

func launchGenDropWord(toks []launchGenTok, word string) []launchGenTok {
	if len(toks) > 0 && toks[0].text == word {
		return toks[1:]
	}
	return toks
}

// launchGenQualPath reads ident (:: ident)* and returns the identifiers, or nil for any other shape.
func launchGenQualPath(toks []launchGenTok) []string {
	var path []string
	for j, t := range toks {
		if j%2 == 0 {
			if t.kind != launchGenIdent {
				return nil
			}
			path = append(path, t.text)
		} else if t.text != "::" {
			return nil
		}
	}
	return path
}

func launchGenHasWord(toks []launchGenTok, word string) bool {
	for _, t := range toks {
		if t.kind == launchGenIdent && t.text == word {
			return true
		}
	}
	return false
}

func launchGenHasPunct(toks []launchGenTok, p string) bool {
	for _, t := range toks {
		if t.kind == launchGenPunct && t.text == p {
			return true
		}
	}
	return false
}

var launchGenIntWords = map[string]bool{
	"int": true, "unsigned": true, "signed": true, "short": true, "long": true, "char": true,
	"size_t": true, "ssize_t": true, "ptrdiff_t": true, "intptr_t": true, "uintptr_t": true,
	"int8_t": true, "int16_t": true, "int32_t": true, "int64_t": true,
	"uint8_t": true, "uint16_t": true, "uint32_t": true, "uint64_t": true, "auto": true,
}

var launchGenDeviceWords = []string{"__constant__", "__device__", "__shared__", "__managed__", "extern", "thread_local"}

// launchGenKindOf classifies a declaration by its type words: integral and dim3 non-pointer values can be
// constants; everything else, and any device-resident or pointer declaration, is unknown.
func launchGenKindOf(base []launchGenTok, ptr bool) (launchGenDeclKind, bool) {
	dim := launchGenHasWord(base, "dim3")
	integral := dim
	for _, t := range base {
		if t.kind == launchGenIdent && launchGenIntWords[t.text] {
			integral = true
		}
	}
	ptr = ptr || launchGenHasPunct(base, "*") || launchGenHasPunct(base, "&")
	device := false
	for _, w := range launchGenDeviceWords {
		device = device || launchGenHasWord(base, w)
	}
	switch {
	case ptr || device || !integral:
		return launchGenDeclUnknown, false
	case launchGenHasWord(base, "const") || launchGenHasWord(base, "constexpr"):
		return launchGenDeclConst, dim
	}
	return launchGenDeclMutable, dim
}

func launchGenIsTypeSpec(base []launchGenTok) bool {
	if len(base) == 0 {
		return false
	}
	ident := false
	for _, t := range base {
		switch t.kind {
		case launchGenIdent:
			ident = true
		case launchGenPunct:
			switch t.text {
			case "*", "&", "::", "<", ">", ",":
			default:
				return false
			}
		default:
			return false
		}
	}
	return ident && base[len(base)-1].text != "::"
}

// launchGenDeclStop returns the index of the token that ends a declarator's name: an initializer, a
// constructor argument list, or an array bound.
func launchGenDeclStop(head []launchGenTok) int {
	for j, t := range head {
		if t.kind == launchGenPunct && (t.text == "=" || t.text == "(" || t.text == "{" || t.text == "[") {
			return j
		}
	}
	return len(head)
}

// launchGenInitOf returns the initializer tokens of a declarator and whether they are a constructor argument
// list written as name(args) or name{args}.
func launchGenInitOf(rem []launchGenTok) ([]launchGenTok, bool) {
	if len(rem) < 2 {
		return nil, false
	}
	switch rem[0].text {
	case "=":
		return rem[1:], false
	case "(", "{":
		return rem[1 : len(rem)-1], true
	}
	return nil, false
}

func launchGenIsAssignOp(s string) bool {
	switch s {
	case "=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=":
		return true
	}
	return false
}

// launchGenIsControl names the keywords that introduce a statement or block, which are never function names.
func launchGenIsControl(s string) bool {
	switch s {
	case "if", "while", "for", "switch", "catch", "do", "else", "return", "sizeof", "decltype":
		return true
	}
	return false
}

// launchGenBuildIndex walks every source in order and returns the index of their declarations and launches.
func launchGenBuildIndex(srcs []launchGenSource) (*launchGenIndex, error) {
	idx := launchGenNewIndex()
	for _, s := range srcs {
		text := blankCommentsAndStrings(strings.ReplaceAll(s.text, "\\\n", "  "))
		w := &launchGenWalker{idx: idx, file: s.file, text: text, toks: launchGenTokenize(text),
			cur: launchGenNewFrame(launchGenFrameNS, nil, nil)}
		if err := w.run(); err != nil {
			return nil, fmt.Errorf("%s: %w", s.file, err)
		}
	}
	return idx, nil
}
