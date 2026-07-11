# Dockerfile — the fak Hugging Face Space (Docker SDK).
#
# Self-contained: the builder git-clones the PUBLIC fak repo and compiles the
# three pure-Go witnesses this Space runs (fak, deletioncert, turntaxdemo). So a
# skeptic can reproduce the whole image from source with nothing but this file —
# the Space repo needs only {Dockerfile, README.md, app.py}. Mirrors the root
# Dockerfile's builder (golang:1.26, CGO_ENABLED=0, -trimpath -s -w).
#
#   docker build -t fak-hf-demo spaces/hf-demo
#   docker run --rm -p 7860:7860 fak-hf-demo   # open http://localhost:7860
#
# Pin a release instead of main:  --build-arg FAK_REF=v0.37.0

# --- builder -------------------------------------------------------------------
FROM golang:1.26 AS build
ARG FAK_REPO=https://github.com/anthony-chaudhary/fak
ARG FAK_REF=main
ARG APP_VERSION=hf-space
ENV CGO_ENABLED=0 GOTOOLCHAIN=auto
WORKDIR /src
# Shallow clone of the exact ref (branch or tag). Zero external Go deps means no
# go.sum fetch step — the module builds offline once cloned.
RUN git clone --depth 1 --branch "${FAK_REF}" "${FAK_REPO}" fak
WORKDIR /src/fak
RUN go build -trimpath \
      -ldflags "-s -w -X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion=${APP_VERSION}" \
      -o /out/fak ./cmd/fak \
 && go build -trimpath -ldflags "-s -w" -o /out/deletioncert ./cmd/deletioncert \
 && go build -trimpath -ldflags "-s -w" -o /out/turntaxdemo ./cmd/turntaxdemo

# --- runtime -------------------------------------------------------------------
# Python (not distroless) because the front door is a Gradio app. The proofs are
# still the Go binaries above; Gradio only shells out to them and renders stdout.
FROM python:3.12-slim
RUN pip install --no-cache-dir "gradio>=5,<6"

# Run as a non-root user (HF Spaces run as uid 1000; be explicit and own the app dir).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# The three witnesses, plus the data they read from a repo checkout: the policy
# manifest (fak preflight) and the turn-tax fixtures (turntaxdemo). Copied to the
# SAME relative paths the commands expect from the repo root.
COPY --from=build --chown=user:user /out/fak /out/deletioncert /out/turntaxdemo ./
COPY --from=build --chown=user:user /src/fak/examples ./examples
COPY --from=build --chown=user:user /src/fak/testdata/turntax ./testdata/turntax

# The Gradio front door (from the Space build context, not the clone).
COPY --chown=user:user app.py ./

EXPOSE 7860
CMD ["python", "app.py"]
