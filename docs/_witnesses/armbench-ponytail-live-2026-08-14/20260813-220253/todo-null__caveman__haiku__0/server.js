const http = require('http');

const todos = [];
let nextId = 1;
const MAX_BODY_BYTES = 1024 * 1024;

function sendJson(response, statusCode, value) {
  const body = JSON.stringify(value);
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  response.end(body);
}

function findTodo(id) {
  return todos.find((todo) => todo.id === id);
}

function readJson(request, response, callback) {
  const chunks = [];
  let size = 0;
  let rejected = false;

  request.on('data', (chunk) => {
    if (rejected) return;

    size += chunk.length;
    if (size > MAX_BODY_BYTES) {
      rejected = true;
      sendJson(response, 413, { error: 'Request body too large' });
      return;
    }

    chunks.push(chunk);
  });

  request.on('end', () => {
    if (rejected) return;

    try {
      callback(JSON.parse(Buffer.concat(chunks).toString('utf8')));
    } catch {
      sendJson(response, 400, { error: 'Invalid JSON' });
    }
  });

  request.on('error', () => {
    if (!response.headersSent) {
      sendJson(response, 400, { error: 'Invalid request body' });
    }
  });
}

const server = http.createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  const pathname = url.pathname;

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
        sendJson(response, 400, { error: 'Title is required' });
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

    if (request.method === 'GET') {
      const todo = findTodo(id);
      if (!todo) {
        sendJson(response, 404, { error: 'Todo not found' });
        return;
      }

      sendJson(response, 200, todo);
      return;
    }

    if (request.method === 'DELETE') {
      const index = todos.findIndex((todo) => todo.id === id);
      if (index === -1) {
        sendJson(response, 404, { error: 'Todo not found' });
        return;
      }

      todos.splice(index, 1);
      response.writeHead(204);
      response.end();
      return;
    }
  }

  sendJson(response, 404, { error: 'Not found' });
});

server.listen(process.env.PORT || 3000);
