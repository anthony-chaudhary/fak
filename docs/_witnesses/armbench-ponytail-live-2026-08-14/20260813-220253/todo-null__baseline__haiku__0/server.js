const http = require('http');

const todos = [];
let nextId = 1;
const maxBodySize = 1024 * 1024;

function sendJson(response, statusCode, value) {
  const body = JSON.stringify(value);
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  response.end(body);
}

function sendStatus(response, statusCode) {
  response.writeHead(statusCode);
  response.end();
}

function readJson(request, response, onBody) {
  const declaredSize = Number(request.headers['content-length']);
  if (Number.isFinite(declaredSize) && declaredSize > maxBodySize) {
    sendStatus(response, 413);
    request.resume();
    return;
  }

  const chunks = [];
  let size = 0;
  let tooLarge = false;

  request.on('data', (chunk) => {
    if (tooLarge) return;

    size += chunk.length;
    if (size > maxBodySize) {
      tooLarge = true;
      chunks.length = 0;
      return;
    }

    chunks.push(chunk);
  });

  request.on('end', () => {
    if (tooLarge) {
      sendStatus(response, 413);
      return;
    }

    try {
      onBody(JSON.parse(Buffer.concat(chunks).toString('utf8')));
    } catch {
      sendStatus(response, 400);
    }
  });

  request.on('error', () => {
    if (!response.headersSent) sendStatus(response, 400);
  });
}

const server = http.createServer((request, response) => {
  let pathname;
  try {
    pathname = new URL(request.url, 'http://localhost').pathname;
  } catch {
    sendStatus(response, 400);
    return;
  }

  if (request.method === 'GET' && pathname === '/todos') {
    sendJson(response, 200, todos);
    return;
  }

  if (request.method === 'POST' && pathname === '/todos') {
    readJson(request, response, (body) => {
      if (
        body === null ||
        typeof body !== 'object' ||
        Array.isArray(body) ||
        typeof body.title !== 'string' ||
        body.title.trim() === ''
      ) {
        sendStatus(response, 400);
        return;
      }

      const todo = { id: nextId++, title: body.title, done: false };
      todos.push(todo);
      sendJson(response, 201, todo);
    });
    return;
  }

  const match = pathname.match(/^\/todos\/(\d+)$/);
  if (match) {
    const id = Number(match[1]);
    const index = todos.findIndex((todo) => todo.id === id);

    if (request.method === 'GET') {
      if (index === -1) {
        sendStatus(response, 404);
      } else {
        sendJson(response, 200, todos[index]);
      }
      return;
    }

    if (request.method === 'DELETE') {
      if (index === -1) {
        sendStatus(response, 404);
      } else {
        todos.splice(index, 1);
        sendStatus(response, 204);
      }
      return;
    }
  }

  sendStatus(response, 404);
});

server.listen(process.env.PORT || 3000);
