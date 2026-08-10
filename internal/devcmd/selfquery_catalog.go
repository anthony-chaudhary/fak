package devcmd

import (
	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

// loadSelfQueryDevCatalog adapts repository ownership data at the fak-dev boundary.
// internal/selfquery remains runtime-safe and knows only these data transfer records.
func loadSelfQueryDevCatalog(root string) (*selfquery.DevCatalog, error) {
	cat, err := devindex.Load(root)
	if err != nil {
		return nil, err
	}
	out := &selfquery.DevCatalog{}
	for _, leaf := range cat.Leaves {
		out.Leaves = append(out.Leaves, selfquery.DevLeaf{Name: leaf.Name, Tree: leaf.Tree, Desc: leaf.Desc})
	}
	for _, doc := range cat.Docs {
		out.Docs = append(out.Docs, selfquery.DevDoc{Title: doc.Title, Path: doc.Path, Blurb: doc.Blurb})
	}
	for _, claim := range cat.Claims {
		out.Claims = append(out.Claims, selfquery.DevClaim{Tag: claim.Tag, Text: claim.Text, Lanes: append([]string(nil), claim.Lanes...)})
	}
	for _, verb := range cat.Verbs() {
		out.Verbs = append(out.Verbs, selfquery.DevVerb{Name: verb.Name, Synopsis: verb.Synopsis, Lane: verb.Lane, Aliases: append([]string(nil), verb.Aliases...)})
	}
	return out, nil
}
