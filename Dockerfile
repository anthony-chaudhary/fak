# Dockerfile — put `fak` in front of your model in production.
#
# Two stages. The builder compiles the pure-Go `cmd/fak` static (CGO_ENABLED=0), so
# the final image is the distroless `static` base + one ~13 MB binary — no shell, no
# package manager, no libc, runs as nonroot. This is the "static-binary / Docker"
# adopter path of issue #133.
#
#   docker build -t fak .
#   docker run --rm -p 8080:8080 fak serve --addr 0.0.0.0:8080 \
#       --base-url http://host.docker.internal:11434/v1 --model qwen2.5:1.5b
#
# Override APP_VERSION at build time to stamp a specific version into the binary:
#   docker build --build-arg APP_VERSION=0.24.0 -t fak:0.24.0 .

# --- builder -------------------------------------------------------------------
# Cross-compiles for the TARGET platform natively: the toolchain runs on the build
# host's platform (`--platform=$BUILDPLATFORM`) and Go emits the $TARGETARCH binary,
# so the arm64 image is produced by native cross-compilation on the amd64 CI runner
# — never under QEMU emulation. The runtime stage below stays on the target platform.
FROM --platform=$BUILDPLATFORM golang:1.26 AS build
ARG APP_VERSION=docker
ARG TARGETARCH
ENV CGO_ENABLED=0 GOTOOLCHAIN=auto GOARCH=$TARGETARCH
WORKDIR /src
# Copy the module (the Go module is the repo root) and build it. The dependency set
# is two golang.org/x modules pinned by a 4-line go.sum, so the fetch is small enough
# that there is no separate download-cache stage.
COPY . .
# Build through the one canonical release recipe (scripts/build.sh, #3709); this
# stage's ENV (CGO_ENABLED=0, GOTOOLCHAIN=auto, GOARCH=$TARGETARCH) passes through
# to it unchanged — build.sh is toolchain-agnostic and overrides none of them.
RUN OUT=/out/fak VERSION="${APP_VERSION}" sh scripts/build.sh

# --- runtime -------------------------------------------------------------------
FROM gcr.io/distroless/static-debian12:nonroot
# Ownership label for the Official MCP Registry (modelcontextprotocol/registry): when
# this image is published as an `oci` package in server.json, the registry verifies the
# publisher by matching this label to the server name in the io.github.* namespace it
# already authenticated via GitHub. See docs/fak/mcp-registry.md.
LABEL io.modelcontextprotocol.server.name="io.github.anthony-chaudhary/fak"
COPY --from=build /out/fak /usr/local/bin/fak
# Notice files ship inside the image so redistributors can satisfy the
# license notice terms (#14447).
COPY LICENSE NOTICE.md THIRD_PARTY_NOTICES.md /usr/share/doc/fak/
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/fak"]
# Default to the gateway bound to all interfaces (containers need 0.0.0.0, not
# loopback). Override the whole command to run `agent`, `policy`, etc.
CMD ["serve", "--addr", "0.0.0.0:8080"]
