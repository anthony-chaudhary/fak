package main

import (
	"context"
	_ "embed"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

//go:embed index.html
var pageHTML string

var page = template.Must(template.New("page").Parse(pageHTML))

type event struct {
	Type   string
	Detail string
}

type pageData struct {
	Prompt string
	Events []event
	Error  string
}

// app is the tiny product seam: HTTP owns presentation while harnesskit owns
// the portable product description that a real fak host can inspect and grant.
type app struct {
	product harnesskit.Product
}

func newApp() (app, error) {
	product, err := harnesskit.New("example/custom-harness-web", harnesskit.ContractVersion).
		WithProfile(harnesskit.Profile{ID: "offline-learning"}).
		WithTransport(harnesskit.Transport{
			ID: "net-http",
			Provenance: harnesskit.Provenance{
				Source:  "Go standard library net/http",
				Version: "stdlib",
			},
		}).
		Build()
	if err != nil {
		return app{}, err
	}
	return app{product: product}, nil
}

func (a app) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /", a.home)
	mux.HandleFunc("POST /turn", a.turn)
	return mux
}

func (a app) home(w http.ResponseWriter, _ *http.Request) {
	a.render(w, http.StatusOK, pageData{})
}

func (a app) turn(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		a.render(w, http.StatusBadRequest, pageData{Error: "Could not read the form."})
		return
	}
	prompt := strings.TrimSpace(r.FormValue("prompt"))
	if prompt == "" {
		a.render(w, http.StatusBadRequest, pageData{Error: "Write a prompt first."})
		return
	}

	events, err := a.runOfflineTurn(r.Context(), prompt)
	if err != nil {
		a.render(w, http.StatusInternalServerError, pageData{Prompt: prompt, Error: err.Error()})
		return
	}
	a.render(w, http.StatusOK, pageData{Prompt: prompt, Events: events})
}

// runOfflineTurn is deterministic so learners can run and test the complete
// UI-to-harness path without a provider key, network call, or GPU.
func (a app) runOfflineTurn(ctx context.Context, prompt string) ([]event, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	spec := a.product.Spec()
	return []event{
		{Type: "turn.started", Detail: prompt},
		{Type: "model.response", Detail: fmt.Sprintf("%s received %q", spec.ID, prompt)},
		{Type: "tool.requested", Detail: "record_learning_example"},
		{Type: "tool.completed", Detail: "record_learning_example:ok"},
		{Type: "turn.completed", Detail: "ok"},
	}, nil
}

func (a app) render(w http.ResponseWriter, status int, data pageData) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(status)
	if err := page.Execute(w, data); err != nil {
		// Headers may already be written; log-free example keeps the failure visible.
		_, _ = io.WriteString(w, "template error")
	}
}

func main() {
	a, err := newApp()
	if err != nil {
		panic(err)
	}
	fmt.Println("web harness listening on http://127.0.0.1:8080")
	if err := http.ListenAndServe("127.0.0.1:8080", a.routes()); err != nil {
		panic(err)
	}
}
