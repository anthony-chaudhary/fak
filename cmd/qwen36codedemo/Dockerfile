FROM golang:1.26-bookworm AS build
WORKDIR /src
COPY go.mod ./
COPY cmd/qwen36codedemo ./cmd/qwen36codedemo
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /out/qwen36codedemo ./cmd/qwen36codedemo
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/qwen36codedemo /qwen36codedemo
ENTRYPOINT ["/qwen36codedemo"]