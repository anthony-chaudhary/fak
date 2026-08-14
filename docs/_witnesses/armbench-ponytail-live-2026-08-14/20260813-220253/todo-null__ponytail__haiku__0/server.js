const http = require('http');

const todos = [];
let nextId = 1;

function sendJson(response, statusCode, value) {
  const body = JSON.stringify(value);
  response.writeHead(statusCode, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  response.end(body);
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let tooLarge = false;

    request.on('data', (chunk) => {
      if (tooLarge) return;

      size += chunk.length;
      if (size > 1_000_000) {
        tooLarge = true;
        reject(Object.assign(new Error('Request body too large'), { statusCode: 413 }));
        return;
      }

      chunks.push(chunk);
    });

    request.on('end', () => {
      if (tooLarge) return;

      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
      } catch {
        reject(Object.assign(new Error('Invalid JSON'), { statusCode: 400 }));
      }
    });

    request.on('error', reject);
  });
}

const server = http.createServer(async (request, response) => {
  const pathname = request.url.split('?', 1)[0];

  if (request.method === 'GET' && pathname === '/todos') {
    sendJson(response, 200, todos);
    return;
  }

  if (request.method === 'POST' && pathname === '/todos') {
    try {
      const body = await readJson(request);
      if (
        body === null ||
        typeof body !== 'object' ||
        typeof body.title !== 'string' ||
        body.title.trim() === ''
      ) {
        sendJson(response, 400, { error: 'A non-empty title is required' });
        return;
      }

      const todo = { id: nextId++, title: body.title, done: false };
      todos.push(todo);
      sendJson(response, 201, todo);
    } catch (error) {
      sendJson(response, error.statusCode || 400, { error: error.message });
    }
    return;
  }

  const match = /^\/todos\/(\d+)$/.exec(pathname);
  if (match && (request.method === 'GET' || request.method === 'DELETE')) {
    const id = Number(match[1]);
    const index = todos.findIndex((todo) => todo.id === id);

    if (index === -1) {
      sendJson(response, 404, { error: 'Todo not found' });
      return;
    }

    if (request.method === 'GET') {
      sendJson(response, 200, todos[index]);
    } else {
      todos.splice(index, 1);
      response.writeHead(204);
      response.end();
    }
    return;
  }

  sendJson(response, 404, { error: 'Not found' });
});

server.listen(process.env.PORT || 3000);
