#!/usr/bin/env python3
"""HTTP contract tests for the zero-dependency benchmark endpoint."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import bench_endpoint_server as endpoint


def _request(url, *, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.headers.get_content_type(), response.read()


def test_models_health_and_chat_contract():
    server = ThreadingHTTPServer(("127.0.0.1", 0), endpoint.Handler)
    server.model = "contract-model"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, content_type, body = _request(base + "/healthz")
        assert (status, content_type, body) == (200, "text/plain", b"ok")

        _, _, body = _request(base + "/v1/models")
        models = json.loads(body)
        assert models["data"] == [
            {"id": "contract-model", "object": "model", "owned_by": "fleet-bench"}
        ]

        _, _, body = _request(
            base + "/v1/chat/completions",
            body={
                "model": "requested-model",
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello benchmark"},
                ],
            },
        )
        chat = json.loads(body)
        assert chat["model"] == "requested-model"
        assert "hello benchmark" in chat["choices"][0]["message"]["content"]
        assert chat["usage"]["prompt_tokens"] == 4
        assert chat["usage"]["total_tokens"] >= chat["usage"]["prompt_tokens"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_unknown_route_is_structured_404():
    server = ThreadingHTTPServer(("127.0.0.1", 0), endpoint.Handler)
    server.model = "contract-model"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/missing"
        try:
            _request(url)
            raise AssertionError("unknown route unexpectedly succeeded")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
            assert json.loads(exc.read())["error"]["message"] == "no route /missing"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

