// Package docrender turns this repo's Markdown into a print-ready HTML page and,
// behind the same verb, into a PDF — with no new module dependency, and with the
// browser wrapped rather than handed to a human.
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate. In
// practice it imports nothing but the standard library, which is the point.
//
// # Why hand-rolled
//
// fak's entire external dependency set is two golang.org/x extended-standard-library
// modules pinned by a four-line go.sum, and that number is a stated product claim,
// not an accident. A Markdown library, a Node toolchain or a pandoc invocation
// would each buy this package a full CommonMark implementation and cost the claim.
// So the parser is hand-rolled against exactly the constructs this corpus uses. The
// subset is small on purpose, it is enumerated below, every entry has a test in
// subset_test.go, and TestNoNewModuleDependencies asserts go.mod's require set has
// not moved.
//
// # The supported subset
//
// Block constructs. Anything else is refused by name and line — see "Refusal", below:
//
//	# Title              the document title (H1)
//	## Section           a section; under KindDeck it starts a new slide
//	### Subsection       a subsection (H3)
//	paragraph text       consecutive non-blank lines join into one <p>
//	- item / * item      an unordered list item (top level only)
//	1. item              an ordered list item (top level only)
//	- [ ] / - [x] item   a task list item, rendered with a ballot-box glyph
//	> quoted             a blockquote
//	| a | b |            a table row; a |---|---| row separates head from body
//	```                  a fenced code block (which must be closed)
//	---                  a thematic break; under KindDeck a slide separator, dropped
//	![alt](path)         a figure on a line of its own, inlined by InlineFigures
//	<!-- note -->        a whole-line comment: metadata, dropped from the output
//
// Inline constructs, applied inside every block above except the code fence:
//
//	`code`               a code span, extracted first so its contents are never markup
//	[text](url)          a link
//	**bold**             strong
//	*italic*             emphasis
//
// # Refusal, not passthrough
//
// A construct outside that list is a hard error naming the construct and the line
// (see Unsupported and UnsupportedError). It is never rendered as literal text.
//
// That is the whole difference between a bounded parser and a broken one. A `####`
// heading, a nested bullet or a `[ref][1]` link that silently reaches the page as
// its own source text does not look like an unsupported construct — it looks like
// the author typed something wrong, so the bug is filed against the document and
// the renderer keeps its reputation. Scan enumerates the refusals; Parse returns
// them all at once, so one pass over a document fixes it rather than N.
//
// # Kind is an input, never an inference
//
// Kind selects page geometry and what `##` and `---` mean:
//
//	                Deck                       Report                Brief
//	`##`            a new 16:9 slide           a heading in flow     a heading in flow
//	`---`           slide separator, dropped   a horizontal rule     a horizontal rule
//	page            13.333in x 7.5in           8.5in x 11in          8.5in x 11in
//	contents        never                      on request (TOC)      never
//	density         presentation               document              one-pager
//
// Parse REQUIRES a Kind; the zero value is an error, not a default. ResolveKind
// picks one for a caller that has no explicit answer, in this order:
//
//  1. an operator override (a --kind flag), because a human who just said what
//     this is outranks everything;
//  2. an explicit `<!-- kind: deck -->` marker in the document's first 40 lines,
//     because that is the author's own statement;
//  3. a corpus rule keyed on the document's PATH (see kindByPath);
//  4. KindReport, the corpus default.
//
// Content-based inference is deliberately absent, and this is the expensive lesson
// the package is built around. "Mostly short sections, so it is probably a deck"
// was tried upstream and rejected: an explainer with terse sections silently became
// a forty-page landscape deck, and the operator saw a *rendering* bug — huge type,
// one paragraph per page — with no hint that the real fault was a classification
// made three layers earlier. A misclassification that presents as a rendering bug
// is the expensive kind, because the evidence points away from the cause. Path and
// marker are both things a human wrote down on purpose and can therefore correct;
// a heuristic over the body is neither. ResolveKind returns a Decision carrying the
// rule that fired, so `fak docrender kind` can always answer "why is this a deck?".
//
// # The browser is wrapped, not documented
//
// PDF needs a print engine, and the honest options are all vendor binaries. The
// invocation is wrapped here (see FindBrowser and PrintPDF) rather than written
// into a runbook: a README step that says "now run Chrome with these flags" is an
// unmanaged surface — untested, undiscoverable, and unable to fail with a sentence.
// When no browser is installed the verb says so, names what to install and what
// environment variable overrides the search, and still produces the HTML. HTMLOnly
// renders with no browser at all, which is what makes the parser useful in CI.
//
// # Determinism
//
// The HTML this package emits is a pure function of the Markdown, the figures and
// the Kind — there is no clock in it. That is what lets Lock decide re-rendering by
// hashing the INPUT: a print engine never emits byte-identical PDFs twice (it
// stamps a wall-clock creation date), so "did the output change?" cannot be asked
// of the output. Callers that want a provenance line pass Bundle.Stamp explicitly,
// and a Stamp carrying a timestamp defeats the lock by design — it makes every run
// a fresh input. Bundle.Write writes nothing at all when the input hash is unchanged.
//
// # Invariants
//
// Invariant: doc rendering is fail-closed and deterministic.
// Guard condition: Any unsupported Markdown construct is rejected with an explicit
// UnsupportedError naming line and construct; unparsed constructs never pass through
// as unformatted literal text.
// Guard condition: Document Kind is never guessed via content heuristics; classification
// strictly adheres to the precedence chain: explicit override > metadata marker >
// path rule > corpus default.
package docrender
