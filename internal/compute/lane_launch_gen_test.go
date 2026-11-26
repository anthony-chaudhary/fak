package compute

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// This file is the test-only generator for lane_launch_sites.txt. TestLaunchSitesGenerated resolves every
// launch in the package's CUDA sources and compares the rendered table with the checked-in file. Setting
// FAK_LAUNCH_SITES_UPDATE=1 rewrites the file instead. The resolver itself lives in
// lane_launch_gen_resolve_test.go.

// launchGenUpdateEnv, when set to 1, makes TestLaunchSitesGenerated rewrite the data file.
const launchGenUpdateEnv = "FAK_LAUNCH_SITES_UPDATE"

// launchGenDataLocal is the data file as the package-directory tests see it.
const launchGenDataLocal = "lane_launch_sites.txt"

var (
	launchGenErrUnresolved = errors.New("thread count is not a compile-time integer")
	launchGenErrTemplate   = errors.New("thread count depends on a template parameter")
)

// launchGenSiteRes is the resolution of one launch occurrence.
type launchGenSiteRes struct {
	class string
	tpb   int
}

// launchGenEval evaluates thread expressions against an index. active and busy guard against cycles
// through declarations and macros; depth bounds nesting.
type launchGenEval struct {
	idx    *launchGenIndex
	active []*launchGenDecl
	busy   map[string]bool
	depth  int
}

const launchGenMaxDepth = 64

func launchGenNewEval(idx *launchGenIndex) *launchGenEval {
	return &launchGenEval{idx: idx, busy: map[string]bool{}}
}

func launchGenBool(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// evalTokens evaluates a whole token sequence as one expression at ctx.
func (ev *launchGenEval) evalTokens(toks []launchGenTok, ctx launchGenCtx) (launchGenVal, error) {
	if len(toks) == 0 {
		return launchGenVal{}, fmt.Errorf("%w: empty expression", launchGenErrUnresolved)
	}
	if ev.depth >= launchGenMaxDepth {
		return launchGenVal{}, fmt.Errorf("%w: nesting deeper than %d", launchGenErrUnresolved, launchGenMaxDepth)
	}
	ev.depth++
	defer func() { ev.depth-- }()
	p := &launchGenParser{ev: ev, ctx: ctx, toks: toks}
	v, err := p.ternary()
	if err != nil {
		return launchGenVal{}, err
	}
	if p.pos != len(toks) {
		return launchGenVal{}, fmt.Errorf("%w: unparsed tokens from %q", launchGenErrUnresolved, toks[p.pos].text)
	}
	return v, nil
}

// launchGenParser is a recursive-descent reader of C integer expressions with dim3 values.
type launchGenParser struct {
	ev   *launchGenEval
	ctx  launchGenCtx
	toks []launchGenTok
	pos  int
}

func (p *launchGenParser) peek() string {
	if p.pos < len(p.toks) && p.toks[p.pos].kind == launchGenPunct {
		return p.toks[p.pos].text
	}
	return ""
}

func (p *launchGenParser) ternary() (launchGenVal, error) {
	cond, err := p.binary(1)
	if err != nil || p.peek() != "?" {
		return cond, err
	}
	p.pos++
	a, err := p.ternary()
	if err != nil {
		return a, err
	}
	if p.peek() != ":" {
		return launchGenVal{}, fmt.Errorf("%w: ternary without ':'", launchGenErrUnresolved)
	}
	p.pos++
	b, err := p.ternary()
	if err != nil {
		return b, err
	}
	if cond.dim {
		return launchGenVal{}, fmt.Errorf("%w: dim3 used as a condition", launchGenErrUnresolved)
	}
	if cond.n != 0 {
		return a, nil
	}
	return b, nil
}

var launchGenBinPrec = map[string]int{
	"||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6,
	"<": 7, ">": 7, "<=": 7, ">=": 7, "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10,
}

func (p *launchGenParser) binary(minPrec int) (launchGenVal, error) {
	left, err := p.unary()
	if err != nil {
		return left, err
	}
	for {
		op := p.peek()
		prec, ok := launchGenBinPrec[op]
		if !ok || prec < minPrec {
			return left, nil
		}
		p.pos++
		right, err := p.binary(prec + 1)
		if err != nil {
			return right, err
		}
		if left, err = launchGenApply(op, left, right); err != nil {
			return left, err
		}
	}
}

func launchGenApply(op string, a, b launchGenVal) (launchGenVal, error) {
	if a.dim || b.dim {
		return launchGenVal{}, fmt.Errorf("%w: operator %s applied to dim3", launchGenErrUnresolved, op)
	}
	x, y := a.n, b.n
	var r int64
	switch op {
	case "||":
		r = launchGenBool(x != 0 || y != 0)
	case "&&":
		r = launchGenBool(x != 0 && y != 0)
	case "|":
		r = x | y
	case "^":
		r = x ^ y
	case "&":
		r = x & y
	case "==":
		r = launchGenBool(x == y)
	case "!=":
		r = launchGenBool(x != y)
	case "<":
		r = launchGenBool(x < y)
	case ">":
		r = launchGenBool(x > y)
	case "<=":
		r = launchGenBool(x <= y)
	case ">=":
		r = launchGenBool(x >= y)
	case "<<", ">>":
		if y < 0 || y >= 63 {
			return launchGenVal{}, fmt.Errorf("%w: shift by %d", launchGenErrUnresolved, y)
		}
		if op == "<<" {
			r = x << uint(y)
		} else {
			r = x >> uint(y)
		}
	case "+":
		r = x + y
	case "-":
		r = x - y
	case "*":
		r = x * y
	case "/", "%":
		if y == 0 {
			return launchGenVal{}, fmt.Errorf("%w: division by zero", launchGenErrUnresolved)
		}
		if op == "/" {
			r = x / y
		} else {
			r = x % y
		}
	default:
		return launchGenVal{}, fmt.Errorf("%w: operator %s", launchGenErrUnresolved, op)
	}
	return launchGenVal{n: r}, nil
}

func (p *launchGenParser) unary() (launchGenVal, error) {
	switch op := p.peek(); op {
	case "-", "+", "~", "!":
		p.pos++
		v, err := p.unary()
		if err != nil {
			return v, err
		}
		if v.dim {
			return launchGenVal{}, fmt.Errorf("%w: unary %s applied to dim3", launchGenErrUnresolved, op)
		}
		switch op {
		case "-":
			v.n = -v.n
		case "~":
			v.n = ^v.n
		case "!":
			v.n = launchGenBool(v.n == 0)
		}
		return launchGenVal{n: v.n}, nil
	case "(":
		if n := p.intCastLen(); n > 0 {
			p.pos += n
			return p.unary()
		}
	}
	return p.postfix()
}

// intCastLen returns the token count of a "(int-type-words)" cast prefix at the current position, or 0.
func (p *launchGenParser) intCastLen() int {
	n := 0
	j := p.pos + 1
	for ; j < len(p.toks) && p.toks[j].kind == launchGenIdent; j++ {
		if !launchGenIntWords[p.toks[j].text] && p.toks[j].text != "const" {
			return 0
		}
		n++
	}
	if n == 0 || j >= len(p.toks) || p.toks[j].text != ")" {
		return 0
	}
	return n + 2
}

func (p *launchGenParser) postfix() (launchGenVal, error) {
	v, err := p.primary()
	if err != nil {
		return v, err
	}
	for p.peek() == "." {
		p.pos++
		if p.pos >= len(p.toks) || p.toks[p.pos].kind != launchGenIdent {
			return launchGenVal{}, fmt.Errorf("%w: member access without a name", launchGenErrUnresolved)
		}
		member := p.toks[p.pos].text
		p.pos++
		if !v.dim {
			return launchGenVal{}, fmt.Errorf("%w: member .%s of a value that is not a dim3", launchGenErrUnresolved, member)
		}
		switch member {
		case "x":
			v = launchGenVal{n: v.d[0]}
		case "y":
			v = launchGenVal{n: v.d[1]}
		case "z":
			v = launchGenVal{n: v.d[2]}
		default:
			return launchGenVal{}, fmt.Errorf("%w: dim3 member %s", launchGenErrUnresolved, member)
		}
	}
	return v, nil
}

func (p *launchGenParser) primary() (launchGenVal, error) {
	if p.pos >= len(p.toks) {
		return launchGenVal{}, fmt.Errorf("%w: expression ends early", launchGenErrUnresolved)
	}
	t := p.toks[p.pos]
	switch {
	case t.kind == launchGenNumber:
		p.pos++
		n, err := launchGenIntLiteral(t.text)
		if err != nil {
			return launchGenVal{}, err
		}
		return launchGenVal{n: n}, nil
	case t.kind == launchGenIdent || t.text == "::":
		return p.name()
	case t.kind == launchGenPunct && t.text == "(":
		p.pos++
		v, err := p.ternary()
		if err != nil {
			return v, err
		}
		if p.peek() != ")" {
			return launchGenVal{}, fmt.Errorf("%w: missing ')'", launchGenErrUnresolved)
		}
		p.pos++
		return v, nil
	}
	return launchGenVal{}, fmt.Errorf("%w: unsupported token %q", launchGenErrUnresolved, t.text)
}

var launchGenKeywords = map[string]bool{
	"sizeof": true, "alignof": true, "decltype": true, "new": true, "delete": true, "this": true,
	"nullptr": true, "true": true, "false": true, "static_cast": true, "reinterpret_cast": true,
	"const_cast": true, "operator": true,
}

// name reads a possibly qualified identifier and resolves it as a call or a value.
func (p *launchGenParser) name() (launchGenVal, error) {
	if p.peek() == "::" {
		return launchGenVal{}, fmt.Errorf("%w: rooted qualified name", launchGenErrUnresolved)
	}
	var parts []string
	for {
		if p.pos >= len(p.toks) || p.toks[p.pos].kind != launchGenIdent {
			return launchGenVal{}, fmt.Errorf("%w: qualified name without an identifier", launchGenErrUnresolved)
		}
		parts = append(parts, p.toks[p.pos].text)
		p.pos++
		if p.peek() != "::" {
			break
		}
		p.pos++
	}
	qual, name := parts[:len(parts)-1], parts[len(parts)-1]
	if launchGenKeywords[name] {
		return launchGenVal{}, fmt.Errorf("%w: keyword %s", launchGenErrUnresolved, name)
	}
	if p.peek() == "(" {
		return p.call(qual, name)
	}
	return p.ev.value(p.ctx, qual, name)
}

func (p *launchGenParser) call(qual []string, name string) (launchGenVal, error) {
	p.pos++
	var args []launchGenVal
	for p.peek() != ")" {
		v, err := p.ternary()
		if err != nil {
			return v, err
		}
		args = append(args, v)
		if p.peek() == "," {
			p.pos++
		} else if p.peek() != ")" {
			return launchGenVal{}, fmt.Errorf("%w: call to %s without ')'", launchGenErrUnresolved, name)
		}
	}
	p.pos++
	if len(qual) == 0 && name == "dim3" {
		return launchGenDim3(args)
	}
	return p.ev.callNamed(p.ctx, qual, name, args)
}

// launchGenDim3 builds a dim3 from one to three scalar components; missing components are 1.
func launchGenDim3(args []launchGenVal) (launchGenVal, error) {
	if len(args) == 0 || len(args) > 3 {
		return launchGenVal{}, fmt.Errorf("%w: dim3 with %d components", launchGenErrUnresolved, len(args))
	}
	v := launchGenVal{dim: true, d: [3]int64{1, 1, 1}}
	for i, a := range args {
		if a.dim {
			return launchGenVal{}, fmt.Errorf("%w: dim3 component is a dim3", launchGenErrUnresolved)
		}
		v.d[i] = a.n
	}
	return v, nil
}

// value resolves a name used as a value: every visible macro and declaration must agree.
func (ev *launchGenEval) value(ctx launchGenCtx, qual []string, name string) (launchGenVal, error) {
	var vals []launchGenVal
	var errs []error
	if len(qual) == 0 {
		for _, m := range ev.idx.macrosVisible(ctx.file, ctx.off, name) {
			if m.fnLike {
				errs = append(errs, fmt.Errorf("%w: function-like macro %s used as a value", launchGenErrUnresolved, name))
				continue
			}
			v, err := ev.macroValue(ctx, m)
			if err != nil {
				errs = append(errs, err)
				continue
			}
			vals = append(vals, v)
		}
	}
	if ctx.viaMacro {
		if n := ev.idx.macroEvents(ctx.file, name); n > 1 {
			return launchGenVal{}, fmt.Errorf("%w: %s is defined %d times in %s; a macro body cannot be placed at its expansion", launchGenErrUnresolved, name, n, ctx.file)
		}
		if len(vals)+len(errs) == 0 {
			return launchGenVal{}, fmt.Errorf("%w: %s is not a macro in a macro body", launchGenErrUnresolved, name)
		}
		return launchGenAgree(vals, errs, name)
	}
	for _, d := range ev.idx.lookup(ctx, qual, name) {
		v, err := ev.declValue(d)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		vals = append(vals, v)
	}
	if len(vals)+len(errs) == 0 {
		return launchGenVal{}, fmt.Errorf("%w: %s has no visible definition", launchGenErrUnresolved, name)
	}
	return launchGenAgree(vals, errs, name)
}

// callNamed evaluates a call to a function defined in the sources, or refuses it.
func (ev *launchGenEval) callNamed(ctx launchGenCtx, qual []string, name string, args []launchGenVal) (launchGenVal, error) {
	if len(qual) == 0 {
		for _, m := range ev.idx.macrosVisible(ctx.file, ctx.off, name) {
			if m.fnLike {
				return launchGenVal{}, fmt.Errorf("%w: function-like macro %s", launchGenErrUnresolved, name)
			}
		}
	}
	if ctx.viaMacro {
		return launchGenVal{}, fmt.Errorf("%w: call to %s in a macro body", launchGenErrUnresolved, name)
	}
	var vals []launchGenVal
	var errs []error
	for _, d := range ev.idx.lookup(ctx, qual, name) {
		if d.kind != launchGenDeclFunc {
			errs = append(errs, fmt.Errorf("%w: %s is not a function", launchGenErrUnresolved, name))
			continue
		}
		v, err := ev.callDecl(d, args)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		vals = append(vals, v)
	}
	if len(vals)+len(errs) == 0 {
		return launchGenVal{}, fmt.Errorf("%w: no function %s", launchGenErrUnresolved, name)
	}
	return launchGenAgree(vals, errs, name)
}

// callDecl evaluates a function whose body is a single return statement, with its parameters bound to the
// argument values and its template parameters inherited unbound.
func (ev *launchGenEval) callDecl(d *launchGenDecl, args []launchGenVal) (launchGenVal, error) {
	if len(args) != len(d.params) {
		return launchGenVal{}, fmt.Errorf("%w: %s called with %d arguments, takes %d", launchGenErrUnresolved, d.name, len(args), len(d.params))
	}
	n := len(d.body)
	if n < 3 || d.body[0].text != "return" || d.body[n-1].text != ";" {
		return launchGenVal{}, fmt.Errorf("%w: %s body is not a single return", launchGenErrUnresolved, d.name)
	}
	bind := launchGenNewFrame(launchGenFrameFunc, d.inner.parent, d.inner.ns)
	for name, ds := range d.inner.decls {
		if ds[0].kind == launchGenDeclTemplate {
			bind.decls[name] = ds
		}
	}
	for i, p := range d.params {
		if p != "" {
			bind.decls[p] = []*launchGenDecl{{name: p, kind: launchGenDeclPreset, val: args[i]}}
		}
	}
	return ev.evalTokens(d.body[1:n-1], launchGenCtx{file: d.file, off: d.off, frame: bind})
}

func (ev *launchGenEval) macroValue(ctx launchGenCtx, m launchGenMacro) (launchGenVal, error) {
	if ev.busy[m.name] {
		return launchGenVal{}, fmt.Errorf("%w: macro %s expands to itself", launchGenErrUnresolved, m.name)
	}
	ev.busy[m.name] = true
	defer delete(ev.busy, m.name)
	return ev.evalTokens(m.body, ctx)
}

func (ev *launchGenEval) isActive(d *launchGenDecl) bool {
	for _, a := range ev.active {
		if a == d {
			return true
		}
	}
	return false
}

// declValue is the value of a declaration. Unknown, template and function declarations have none; a mutable
// variable has one only in a function where it is never assigned.
func (ev *launchGenEval) declValue(d *launchGenDecl) (launchGenVal, error) {
	switch d.kind {
	case launchGenDeclPreset:
		return d.val, nil
	case launchGenDeclTemplate:
		return launchGenVal{}, fmt.Errorf("%w: %s", launchGenErrTemplate, d.name)
	case launchGenDeclUnknown, launchGenDeclFunc:
		return launchGenVal{}, fmt.Errorf("%w: %s", launchGenErrUnresolved, d.name)
	case launchGenDeclMutable:
		if d.frame.kind != launchGenFrameFunc || d.frame.assigned[d.name] {
			return launchGenVal{}, fmt.Errorf("%w: %s is assigned or not function-local", launchGenErrUnresolved, d.name)
		}
	}
	if ev.isActive(d) {
		return launchGenVal{}, fmt.Errorf("%w: %s depends on itself", launchGenErrUnresolved, d.name)
	}
	ev.active = append(ev.active, d)
	defer func() { ev.active = ev.active[:len(ev.active)-1] }()
	v, err := ev.initValue(d)
	if err != nil {
		return v, err
	}
	if d.dim && !v.dim {
		return launchGenVal{dim: true, d: [3]int64{v.n, 1, 1}}, nil
	}
	if !d.dim && v.dim {
		return launchGenVal{}, fmt.Errorf("%w: %s is not a dim3 but has one as its value", launchGenErrUnresolved, d.name)
	}
	return v, nil
}

func (ev *launchGenEval) initValue(d *launchGenDecl) (launchGenVal, error) {
	ctx := launchGenCtx{file: d.file, off: d.off, frame: d.frame}
	switch {
	case d.kind == launchGenDeclEnum && d.init == nil:
		if d.prev == nil {
			return launchGenVal{}, nil
		}
		pv, err := ev.declValue(d.prev)
		if err != nil {
			return pv, err
		}
		if pv.dim {
			return launchGenVal{}, fmt.Errorf("%w: enumerator after a dim3", launchGenErrUnresolved)
		}
		return launchGenVal{n: pv.n + 1}, nil
	case d.ctorArgs:
		var args []launchGenVal
		for _, piece := range launchGenSplitTop(d.init, ",") {
			v, err := ev.evalTokens(piece, ctx)
			if err != nil {
				return v, err
			}
			args = append(args, v)
		}
		return launchGenDim3(args)
	case d.init == nil:
		return launchGenVal{}, fmt.Errorf("%w: %s has no initializer", launchGenErrUnresolved, d.name)
	}
	return ev.evalTokens(d.init, ctx)
}

// launchGenAgree returns the common value of the candidates. A template-dependent error wins over other
// errors, and any error or disagreement leaves the name unresolved.
func launchGenAgree(vals []launchGenVal, errs []error, what string) (launchGenVal, error) {
	for _, err := range errs {
		if errors.Is(err, launchGenErrTemplate) {
			return launchGenVal{}, err
		}
	}
	if len(errs) > 0 {
		return launchGenVal{}, errs[0]
	}
	for _, v := range vals[1:] {
		if v != vals[0] {
			return launchGenVal{}, fmt.Errorf("%w: definitions of %s disagree", launchGenErrUnresolved, what)
		}
	}
	return vals[0], nil
}

// classify gives the resolution class and thread count of one launch occurrence.
func (ev *launchGenEval) classify(l launchGenLaunch) launchGenSiteRes {
	if n, err := strconv.Atoi(l.scan.ThreadsExpr); err == nil {
		if n <= 0 {
			return launchGenSiteRes{class: LaunchGenResRuntime}
		}
		return launchGenSiteRes{class: LaunchGenResLiteral, tpb: n}
	}
	toks := launchGenTokenize(l.scan.ThreadsExpr)
	v, err := ev.evalTokens(toks, launchGenCtx{file: l.file, off: l.off, frame: l.frame, viaMacro: l.viaMacro})
	if err != nil {
		if errors.Is(err, launchGenErrTemplate) {
			return launchGenSiteRes{class: LaunchGenResTemplate}
		}
		return launchGenSiteRes{class: LaunchGenResRuntime}
	}
	if n := v.threads(); n > 0 {
		return launchGenSiteRes{class: LaunchGenResChain, tpb: int(n)}
	}
	return launchGenSiteRes{class: LaunchGenResRuntime}
}

// launchGenSiteKey is one distinct launch text in one file, the unit of the data table.
type launchGenSiteKey struct {
	file string
	scan scannedLaunch
}

// launchGenCombine gives one row's class: occurrences that agree keep their class; occurrences that disagree
// are unjudged as runtime.
func launchGenCombine(rs []launchGenSiteRes) (string, int) {
	for _, r := range rs[1:] {
		if r != rs[0] {
			return LaunchGenResRuntime, 0
		}
	}
	return rs[0].class, rs[0].tpb
}

func launchGenInventoried(file string) bool {
	ext := path.Ext(file)
	return ext == ".cu" || ext == ".cuh"
}

// launchGenInventory resolves every launch in the .cu and .cuh sources into one row per distinct launch text,
// sorted as the data file is.
func launchGenInventory(srcs []launchGenSource) ([]LaunchSite, error) {
	idx, err := launchGenBuildIndex(srcs)
	if err != nil {
		return nil, err
	}
	ev := launchGenNewEval(idx)
	groups := map[launchGenSiteKey][]launchGenSiteRes{}
	for _, l := range idx.launches {
		if !launchGenInventoried(l.file) {
			continue
		}
		k := launchGenSiteKey{file: l.file, scan: l.scan}
		groups[k] = append(groups[k], ev.classify(l))
	}
	rows := make([]LaunchSite, 0, len(groups))
	for k, rs := range groups {
		class, tpb := launchGenCombine(rs)
		rows = append(rows, LaunchSite{File: k.file, Kernel: k.scan.Kernel, ThreadsExpr: k.scan.ThreadsExpr,
			ThreadsPerBlock: tpb, Resolution: class, GridExpr: k.scan.GridExpr, SmemExpr: k.scan.SmemExpr})
	}
	sort.Slice(rows, func(i, j int) bool { return compareLaunchSites(rows[i], rows[j]) < 0 })
	return rows, nil
}

// launchGenRender writes the data file: the header, then one tab-separated row per launch.
func launchGenRender(rows []LaunchSite) string {
	var b strings.Builder
	b.WriteString(launchSiteDataHeader)
	b.WriteByte('\n')
	for _, s := range rows {
		fmt.Fprintf(&b, "%s\t%s\t%s\t%d\t%s\t%s\t%s\n", s.File, s.Kernel, s.ThreadsExpr, s.ThreadsPerBlock,
			s.Resolution, s.GridExpr, s.SmemExpr)
	}
	return b.String()
}

// launchGenLoadSources reads the CUDA sources and the CUDA headers of the package directory.
func launchGenLoadSources(t *testing.T) []launchGenSource {
	t.Helper()
	var srcs []launchGenSource
	err := filepath.WalkDir(".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		slash := filepath.ToSlash(p)
		if !launchGenIsCUDASource(slash) {
			return nil
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		srcs = append(srcs, launchGenSource{file: path.Join(launchSiteDir, slash), text: string(b)})
		return nil
	})
	if err != nil {
		t.Fatalf("walk %s: %v", launchSiteDir, err)
	}
	sort.Slice(srcs, func(i, j int) bool { return srcs[i].file < srcs[j].file })
	return srcs
}

// launchGenIsCUDASource admits .cu and .cuh files, and the .h files that belong to the CUDA backend, since
// they declare constants the kernels use. Vulkan, Metal and ROCm headers are left out.
func launchGenIsCUDASource(slash string) bool {
	switch path.Ext(slash) {
	case ".cu", ".cuh":
		return true
	case ".h":
		return strings.HasPrefix(path.Base(slash), "cuda") || strings.HasPrefix(slash, "cuda/")
	}
	return false
}

// TestLaunchSitesGenerated regenerates the inventory from source and checks the checked-in file against it.
func TestLaunchSitesGenerated(t *testing.T) {
	rows, err := launchGenInventory(launchGenLoadSources(t))
	if err != nil {
		t.Fatalf("resolve launch sites: %v", err)
	}
	want := launchGenRender(rows)
	if os.Getenv(launchGenUpdateEnv) == "1" {
		if err := os.WriteFile(launchGenDataLocal, []byte(want), 0o644); err != nil {
			t.Fatalf("write %s: %v", launchSiteDataFile, err)
		}
		t.Logf("rewrote %s with %d rows", launchSiteDataFile, len(rows))
		return
	}
	got, err := os.ReadFile(launchGenDataLocal)
	if err != nil {
		t.Fatalf("read %s: %v", launchSiteDataFile, err)
	}
	if string(got) == want {
		return
	}
	gl, wl := strings.Split(string(got), "\n"), strings.Split(want, "\n")
	for i := 0; i < len(gl) && i < len(wl); i++ {
		if gl[i] != wl[i] {
			t.Fatalf("%s line %d differs from the generated table (rows %d): have %q, generated %q; rerun with %s=1",
				launchSiteDataFile, i+1, len(rows), gl[i], wl[i], launchGenUpdateEnv)
		}
	}
	t.Fatalf("%s has %d lines, generated table has %d; rerun with %s=1",
		launchSiteDataFile, len(gl), len(wl), launchGenUpdateEnv)
}

// TestLaunchGenWalkMatchesScan checks that the scope walker finds exactly the launches the text scanner finds
// in each .cu and .cuh file, so the generated table covers the same sites as the coverage test.
func TestLaunchGenWalkMatchesScan(t *testing.T) {
	srcs := launchGenLoadSources(t)
	idx, err := launchGenBuildIndex(srcs)
	if err != nil {
		t.Fatalf("index: %v", err)
	}
	walked := map[string][]string{}
	for _, l := range idx.launches {
		walked[l.file] = append(walked[l.file], l.scan.String())
	}
	for _, s := range srcs {
		if !launchGenInventoried(s.file) {
			continue
		}
		scanned, err := scanLaunches(s.text)
		if err != nil {
			t.Fatalf("scan %s: %v", s.file, err)
		}
		want := launchStrings(scanned)
		got := walked[s.file]
		sort.Strings(want)
		sort.Strings(got)
		if strings.Join(got, "\n") != strings.Join(want, "\n") {
			t.Errorf("%s: walker found %d launches, text scan found %d", s.file, len(got), len(want))
		}
	}
}

func TestLaunchGenIntLiteral(t *testing.T) {
	tests := []struct {
		text string
		want int64
		ok   bool
	}{
		{"128", 128, true}, {"128u", 128, true}, {"1ULL", 1, true}, {"0x80", 128, true},
		{"010", 8, true}, {"0", 0, true}, {"1.5", 0, false}, {"1e3", 0, false}, {"1.0f", 0, false}, {"09", 0, false},
	}
	for _, tt := range tests {
		got, err := launchGenIntLiteral(tt.text)
		if (err == nil) != tt.ok || got != tt.want {
			t.Errorf("launchGenIntLiteral(%q) = %d, %v; want %d, ok=%v", tt.text, got, err, tt.want, tt.ok)
		}
	}
}

// launchGenResolveSnippet classifies every launch in one synthetic source, keyed by kernel name. Each kernel
// name must appear once so the result is unambiguous.
func launchGenResolveSnippet(t *testing.T, src string) map[string]launchGenSiteRes {
	t.Helper()
	idx, err := launchGenBuildIndex([]launchGenSource{{file: launchSiteDir + "/cuda/snippet.cu", text: src}})
	if err != nil {
		t.Fatalf("index snippet: %v", err)
	}
	ev := launchGenNewEval(idx)
	out := map[string]launchGenSiteRes{}
	for _, l := range idx.launches {
		if _, dup := out[l.scan.Kernel]; dup {
			t.Fatalf("kernel %s launched twice in one snippet", l.scan.Kernel)
		}
		out[l.scan.Kernel] = ev.classify(l)
	}
	return out
}

// TestLaunchGenResolve checks the resolver on synthetic snippets, one per chain form and one per refusal.
// Each case names the kernel whose thread count is judged and the class and count it must get.
func TestLaunchGenResolve(t *testing.T) {
	tests := []struct {
		name   string
		src    string
		kernel string
		class  string
		tpb    int
	}{
		{"literal", "void f() { k1<<<g, 128>>>(x); }", "k1", LaunchGenResLiteral, 128},
		{"namespace consts with dim3 product", `
namespace fak_a {
constexpr int warp_size = 32;
static const int nthreads = 128;
void f() { k2<<<g, dim3(warp_size, nthreads / warp_size, 1)>>>(x); }
}`, "k2", LaunchGenResChain, 128},
		{"local dim3 and block.y halving", "void f() { dim3 block(32, 4); dim3 half(block.x, block.y / 2); k3<<<g, block>>>(x); k4<<<g, half>>>(x); }", "k3", LaunchGenResChain, 128},
		{"local dim3 halving", "void f() { dim3 block(32, 4); dim3 half(block.x, block.y / 2); k3<<<g, block>>>(x); k4<<<g, half>>>(x); }", "k4", LaunchGenResChain, 64},
		{"macro chain", "#define FAK_PA_NUM_THREADS 128\n#define NUM_THREADS FAK_PA_NUM_THREADS\nvoid f() { k5<<<g, NUM_THREADS>>>(x); }", "k5", LaunchGenResChain, 128},
		{"macro undefined before use", "#define T5 64\nvoid f() { k6<<<g, T5>>>(x); }\n#undef T5\nvoid h() { k7<<<g, T5>>>(x); }", "k7", LaunchGenResRuntime, 0},
		{"macro defined before use", "#define T5 64\nvoid f() { k6<<<g, T5>>>(x); }\n#undef T5\nvoid h() { k7<<<g, T5>>>(x); }", "k6", LaunchGenResChain, 64},
		{"conflicting namespace definitions refused", "namespace fak_b { constexpr int n6 = 64; }\nnamespace fak_b { constexpr int n6 = 128; }\nvoid f() { k8<<<g, fak_b::n6>>>(x); }", "k8", LaunchGenResRuntime, 0},
		{"agreeing duplicate definitions", "namespace fak_b2 { constexpr int n = 128; }\nnamespace fak_b2 { constexpr int n = 128; }\nvoid f() { k9<<<g, fak_b2::n>>>(x); }", "k9", LaunchGenResChain, 128},
		{"parameter shadows global constant", "constexpr int t10 = 128;\nvoid f(int t10) { k10<<<g, t10>>>(x); }", "k10", LaunchGenResRuntime, 0},
		{"template parameter", "template <int W> void f() { k11<<<g, W * 32>>>(x); }", "k11", LaunchGenResTemplate, 0},
		{"assigned local refused", "int runtime_value();\nvoid f() { int t = 128; t = runtime_value(); k12<<<g, t>>>(x); }", "k12", LaunchGenResRuntime, 0},
		{"unassigned local", "void f() { int t = 128; k13<<<g, t>>>(x); }", "k13", LaunchGenResChain, 128},
		{"macro body with a redefined name refused", "#define T 64\n#define L(F) k_m1<<<1, T>>>(F)\n#undef T\n#define T 128\nvoid f() { L(1); }", "k_m1", LaunchGenResRuntime, 0},
		{"dim3 member assignment refused", "void f() { dim3 block(128); block.x = 64; k_d1<<<1, block>>>(x); }", "k_d1", LaunchGenResRuntime, 0},
		{"dim3 member compound assignment refused", "void f() { dim3 block(32, 8); block.y /= 2; k_d2<<<1, block>>>(x); }", "k_d2", LaunchGenResRuntime, 0},
		{"dim3 member increment refused", "void f() { dim3 block(32, 4); block.x++; k_d3<<<1, block>>>(x); }", "k_d3", LaunchGenResRuntime, 0},
		{"reference alias write refused", "void f() { int t = 128; auto &r = t; r = 64; k_r1<<<1, t>>>(x); }", "k_r1", LaunchGenResRuntime, 0},
		{"out-of-class member wins over namespace constant", "constexpr int threads = 256; struct P1S { static constexpr int threads = 64; void run(); };\nvoid P1S::run() { k_q1<<<1, threads>>>(x); }", "k_q1", LaunchGenResChain, 64},
		{"mutable struct member refused", "struct S14 { int threads = 64; void f() { k14<<<g, threads>>>(x); } };", "k14", LaunchGenResRuntime, 0},
		{"static constexpr struct member", "struct S15 { static constexpr int kN = 96; void f() { k15<<<g, kN>>>(x); } };", "k15", LaunchGenResChain, 96},
		{"runtime parameter", "void f(int n) { k16<<<1, n>>>(x); }", "k16", LaunchGenResRuntime, 0},
		{"single-return helper on a literal", "static int rows(int r) { return r >= 4 ? 512 : 256; }\nvoid f() { k17<<<1, rows(4)>>>(x); }", "k17", LaunchGenResChain, 512},
		{"single-return helper on a template parameter", "static int rows(int r) { return r >= 4 ? 512 : 256; }\ntemplate <int R> void f() { k18<<<1, rows(R)>>>(x); }", "k18", LaunchGenResTemplate, 0},
		{"division by zero refused", "constexpr int z19 = 0;\nvoid f() { k19<<<1, 64 / z19>>>(x); }", "k19", LaunchGenResRuntime, 0},
		{"zero threads not a count", "void f() { k20<<<1, 0>>>(x); }", "k20", LaunchGenResRuntime, 0},
		{"lambda body launch recorded", "void f() { auto l = [&](int q) { k21<<<1, 32>>>(x); k22<<<1, q>>>(x); }; }", "k21", LaunchGenResLiteral, 32},
		{"lambda parameter shadows", "void f() { auto l = [&](int q) { k21<<<1, 32>>>(x); k22<<<1, q>>>(x); }; }", "k22", LaunchGenResRuntime, 0},
		{"for-loop variable assigned by increment", "constexpr int n23 = 128;\nvoid f() { for (int n23 = 0; n23 < 4; n23++) { k23<<<1, n23>>>(x); } }", "k23", LaunchGenResRuntime, 0},
		{"for-loop variable never assigned", "void f() { for (int n24 = 64; n24 < 100;) { k24<<<1, n24>>>(x); } }", "k24", LaunchGenResChain, 64},
		{"using namespace", "namespace fak_c { constexpr int nthreads = 256; }\nusing namespace fak_c;\nvoid f() { k25<<<1, nthreads>>>(x); }", "k25", LaunchGenResChain, 256},
		{"qualified name", "namespace fak_d { constexpr int nthreads = 96; }\nvoid f() { k26<<<1, fak_d::nthreads>>>(x); }", "k26", LaunchGenResChain, 96},
		{"enumerator with implicit successor", "enum { kThreads27 = 64, kNext27 };\nvoid f() { k27<<<1, kNext27>>>(x); }", "k27", LaunchGenResChain, 65},
		{"float constant refused", "constexpr float f28 = 128.0f;\nvoid f() { k28<<<1, (int)f28>>>(x); }", "k28", LaunchGenResRuntime, 0},
		{"function-like macro refused", "#define TPB29(x) (x)\nvoid f() { k29<<<1, TPB29(64)>>>(x); }", "k29", LaunchGenResRuntime, 0},
		{"multiple declarators", "void f() { int a32 = 32, b32 = 4; k32<<<1, a32 * b32>>>(x); }", "k32", LaunchGenResChain, 128},
		{"inner namespace hides outer", "constexpr int n33 = 16;\nnamespace fak_e { constexpr int n33 = 64; void f() { k33<<<1, n33>>>(x); } }", "k33", LaunchGenResChain, 64},
		{"outer namespace visible from nested", "constexpr int base34 = 16;\nnamespace fak_f { namespace fak_g { void f() { k34<<<1, base34 * 2>>>(x); } } }", "k34", LaunchGenResChain, 32},
		{"conflicting macro definitions refused", "#define N35 64\n#define N35 128\nvoid f() { k35<<<1, N35>>>(x); }", "k35", LaunchGenResRuntime, 0},
		{"range-for variable hides outer constant", "constexpr int n36 = 128;\nvoid f(int* xs) { for (int n36 : xs) { k36<<<1, n36>>>(x); } }", "k36", LaunchGenResRuntime, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := launchGenResolveSnippet(t, tt.src)[tt.kernel]
			if !ok {
				t.Fatalf("kernel %s not found in snippet", tt.kernel)
			}
			if got.class != tt.class || got.tpb != tt.tpb {
				t.Errorf("%s = (%s, %d), want (%s, %d)", tt.kernel, got.class, got.tpb, tt.class, tt.tpb)
			}
		})
	}
}

// TestLaunchGenIgnoresCommentedLaunch checks that launch syntax inside a comment is not a site.
func TestLaunchGenIgnoresCommentedLaunch(t *testing.T) {
	got := launchGenResolveSnippet(t, "// k30<<<1, 64>>>(x);\n/* k31<<<1, 64>>>(x); */\nvoid f() { k32<<<1, 32>>>(x); }")
	if len(got) != 1 || got["k32"].tpb != 32 {
		t.Errorf("snippet launches = %v, want only k32 with 32 threads", got)
	}
}
