# txtai-areas

A small document search server built on [txtai](https://github.com/neuml/txtai): one index per **area**,
a semantic graph for graph RAG, and a graph view in the browser.

- **Areas** keep different kinds of documents apart. Names are hierarchical with dots
  (`dev`, `academic.math`, `notes.work`); searching `academic` also searches every `academic.*` area,
  and `exclude` leaves areas (and everything under them) out.
- Hits from several areas are merged by score: all areas use the same embedding model.
- Each index keeps txtai's **semantic graph** (edges between similar chunks, no LLM extraction).
  A search can add the chunks linked to its hits (`graph`).
- Embeddings go through [LiteLLM](https://github.com/BerriAI/litellm), so any embedding API works:
  OpenAI, or a self-hosted OpenAI-compatible server (Ollama, a LiteLLM proxy, ...).
- The **graph view** (`/ui`) searches an area and draws the sections around a hit and how they connect.

The server only stores and searches. Splitting files into chunks and keeping them in step with their source
is up to the client that calls `/upsert` and `/delete`.

## API

Every call except `/health` needs the header `X-API-Key: <API_KEY>`.

| Call | Body | Answer |
|---|---|---|
| `GET /health` | | `{ok, areas}` |
| `GET /areas` | | `{area: number of chunks}` |
| `POST /upsert` | `{area, documents: [{id, text, source}]}` | `{area, count}` |
| `POST /delete` | `{area, prefix}` or `{area, prefixes: [...]}` | `{area, deleted}`: every chunk whose id starts with a prefix. An area left with no chunks is removed (its folder too) and the answer has `removed: true` |
| `POST /search` | `{query, areas?, exclude?, limit?: 5, graph?: 0, min_weight?: 0.75}` | `[{area, id, source, text, score, via?}]` |

`graph: n` adds up to `n` chunks that the graph links to the hits (edges with at least `min_weight`);
they score *hit score × edge weight* and carry `via`, the hit they hang on.

Chunk ids are `<source>:<path>#<section>#<n>`, for example `notes:guide/setup.md#Install#0`.
The graph view groups chunks into sections by cutting off the last `#<n>`, and deleting by the prefix
`notes:guide/setup.md#` removes a whole file.

```sh
curl -s localhost:9630/search -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"query": "how to install", "areas": ["notes"], "limit": 5, "graph": 2}'
```

## Graph view

`/ui` (HTTP Basic auth with `UI_USER` / `UI_PASSWORD`; off while `UI_PASSWORD` is empty) draws with
[Cytoscape.js](https://js.cytoscape.org/) from cdnjs. Search an area, pick a hit, and the page draws that section
and the sections linked to it (**Steps**: 1 or 2 links away, at most 60; **Edge similarity**: the weakest link
followed). Click a section to read its text; double-click it, or **Around this**, to draw around it instead.
A whole area is not drawn: a big one has thousands of sections, too many to lay out in a browser or to read.
Choose the area in a tree on the left (**Tree**, can be hidden with ☰ and filtered by name) or with one drop-down
per level in the header (**Levels**). A level that is not an area and has only one level below is shown joined with
it (`hm.book`), so deep names stay short.
Put the server behind HTTPS (a reverse proxy) if the UI is reachable from other machines.

## Run

```sh
python3 -m venv venv
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # no GPU needed
venv/bin/pip install -r requirements.txt
cp .env.example .env    # set API_KEY, EMBEDDINGS_*, UI_PASSWORD
set -a; . ./.env; set +a
venv/bin/uvicorn server:app --host 0.0.0.0 --port 9630
```

`txtai-areas.service` is an example systemd unit.

| Setting | Notes |
|---|---|
| `API_KEY` | Clients send it in `X-API-Key` |
| `EMBEDDINGS_MODEL` | As LiteLLM names it, e.g. `openai/text-embedding-3-small`, or `openai/<model>` with `EMBEDDINGS_API_BASE` for an OpenAI-compatible server |
| `EMBEDDINGS_API_BASE`, `EMBEDDINGS_API_KEY` | Where and how to call the embedding API |
| `DATA_DIR` | Where the indexes are kept (default `./data`, one folder per area) |
| `UI_USER`, `UI_PASSWORD` | Login of `/ui` |

Changing the embedding model needs every area indexed again: delete `DATA_DIR` and send the documents again.

## License

MIT
