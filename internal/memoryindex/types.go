package memoryindex

import "sort"

const Schema = "fak-memory-index/1"
const (
	KindMissingFromIndex = "missing_from_index"
	KindIndexLineNoFile  = "index_line_with_no_file"
	KindSlugMismatch     = "slug_filename_mismatch"
	KindDuplicateSlug    = "duplicate_slug"
	KindFrontmatter      = "invalid_frontmatter"
	KindTypeVocabulary   = "type_vocabulary_violation"
	KindUnresolvedLink   = "unresolved_wikilink"
)
const (
	FindingMissingFromIndex     = KindMissingFromIndex
	FindingIndexLineWithNoFile  = KindIndexLineNoFile
	FindingSlugFilenameMismatch = KindSlugMismatch
	FindingDuplicateSlug        = KindDuplicateSlug
	FindingInvalidFrontmatter   = KindFrontmatter
	FindingDanglingWikilink     = KindUnresolvedLink
)

func Kinds() []string {
	return []string{KindMissingFromIndex, KindIndexLineNoFile, KindSlugMismatch, KindDuplicateSlug, KindFrontmatter, KindTypeVocabulary, KindUnresolvedLink}
}

type Finding struct {
	Kind, Subject, Where, File, Slug, Target, Detail string
	Tier                                             string
	Line                                             int
}
type Frontmatter struct {
	Name, Description, Type string
	Present, Terminated     bool
}
type Row struct {
	Tier                                  string
	Line                                  int
	Title, Target, Slug, Description, Raw string
}
type File struct {
	Name      string
	Front     Frontmatter
	Wikilinks []string
}
type Store struct {
	Dir            string
	Tiers, Present []string
	Rows           []Row
	Files          []File
}
type Options struct{ Types []string }
type Report struct {
	Schema, Dir string
	Findings    []Finding
	Counts      map[string]int
	Files       int
	Tiers       []string
	Rows        int
}

func (r Report) Drifted() bool  { return r.Gating() > 0 }
func (r Report) HasDrift() bool { return r.Drifted() }
func (r Report) Gating() int {
	n := 0
	for k, v := range r.Counts {
		if k != KindUnresolvedLink {
			n += v
		}
	}
	return n
}
func (r Report) Fixable() int       { return r.Counts[KindMissingFromIndex] + r.Counts[KindIndexLineNoFile] }
func (r Report) Count(k string) int { return r.Counts[k] }
func SortFindings(fs []Finding) {
	sort.Slice(fs, func(i, j int) bool {
		a, b := fs[i], fs[j]
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.Subject != b.Subject {
			return a.Subject < b.Subject
		}
		return a.Where < b.Where
	})
}
