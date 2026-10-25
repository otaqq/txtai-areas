"""txtai-areas: a small search server with one txtai index per area, a semantic graph and a graph view.

Areas keep different kinds of documents apart. Names are hierarchical with dots (dev, academic.math, creative.game,
notes.work ...); asking for "academic" also searches every "academic.*" area.
A search goes to the areas asked for, or to all of them, and the hits are merged by score: every area uses the
same embedding model, so the scores compare.
Each index also keeps a semantic graph (txtai: edges between similar chunks, no LLM extraction): a search can add
the chunks the graph links to its hits (graph RAG), and the UI draws the graph.

API (header X-API-Key: API_KEY):
  GET  /health
  GET  /areas                                       -> {area: number of chunks}
  POST /upsert {area, documents: [{id, text, source}]}
  POST /delete {area, prefix | prefixes: [..]}     -> deletes every chunk whose id starts with (one of) the prefix(es)
  POST /search {query, areas?: [..], exclude?: [..], limit?: 5, graph?: 0}  -> [{area, id, source, text, score, via?}]
                                                    (graph: also add up to that many chunks linked in the graph to the hits)
                                                    (exclude: areas left out, with everything under them)
UI (HTTP Basic auth, UI_USER / UI_PASSWORD; off when UI_PASSWORD is empty):
  GET  /ui                                          -> graph view of one area (Cytoscape.js)
  GET  /ui/areas, /ui/graph/{area}, /ui/section/{area}?key=   -> data for the page
  The graph shows sections; an edge is the closest pair of their chunks (txtai's graph).
  Chunk ids are "<source>:<path>#<section>#<n>" (n: the chunk's number in the section), e.g. "notes:a/b.md#Intro#0".
Run: uvicorn server:app --host 0.0.0.0 --port 9630. Settings come from the environment (see .env.example).
"""
import os
import re
import secrets
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from txtai import Embeddings

DATA = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent / "data"))
API_KEY = os.environ["API_KEY"]
# Embeddings through LiteLLM: any model it can call, e.g. an OpenAI-compatible server ("openai/<model>" + api_base)
CONFIG = {
    "method": "litellm",
    "path": os.environ["EMBEDDINGS_MODEL"],
    "vectors": {"api_base": os.environ.get("EMBEDDINGS_API_BASE") or None, "api_key": os.environ.get("EMBEDDINGS_API_KEY") or None},
    "content": True,
    "graph": {"approximate": True},  # on upsert, only new nodes look for neighbours
}
AREA = re.compile(r"^[a-z][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)*$")  # dev, academic.math, creative.game

app = FastAPI(title="txtai-areas")
_indexes: dict[str, Embeddings] = {}
_lock = threading.Lock()  # txtai indexes are not thread safe; writes are rare and searches take ~0.05 s


def auth(x_api_key: str = Header("")):
    if x_api_key != API_KEY:
        raise HTTPException(401, "bad api key")


basic = HTTPBasic()


def ui_auth(c: HTTPBasicCredentials = Depends(basic)):
    ok_user = secrets.compare_digest(c.username, os.environ.get("UI_USER", ""))
    ok_pass = secrets.compare_digest(c.password, os.environ.get("UI_PASSWORD", ""))
    if not (os.environ.get("UI_PASSWORD") and ok_user and ok_pass):
        raise HTTPException(401, "login required", headers={"WWW-Authenticate": "Basic"})


def index(area: str, create: bool = False) -> Embeddings | None:
    if not AREA.match(area):
        raise HTTPException(400, f"bad area name: {area}")
    if area not in _indexes:
        path = DATA / area
        if path.exists():
            emb = Embeddings()
            emb.load(str(path), config=CONFIG)  # the saved config has no api key; take it from here
            _indexes[area] = emb
        elif create:
            _indexes[area] = Embeddings(**CONFIG)
        else:
            return None
    return _indexes[area]


def like(prefix: str) -> str:
    """The prefix for a SQL LIKE. txtai's SQL does not take "escape", so % and _ stay wildcards here:
    the caller must check the real prefix afterwards (ids_with_prefix)."""
    return prefix.replace("'", "''")


def ids_with_prefix(emb: Embeddings, prefix: str) -> list[str]:
    rows = emb.search(f"select id from txtai where id like '{like(prefix)}%'", limit=100000)
    return [r["id"] for r in rows if r["id"].startswith(prefix)]


def areas() -> list[str]:
    return sorted(p.name for p in DATA.iterdir() if p.is_dir()) if DATA.exists() else []


def under(area: str, names: list[str]) -> bool:
    return any(area == n or area.startswith(n + ".") for n in names)


def expand(asked: list[str] | None, exclude: list[str] | None = None) -> list[str]:
    """Areas to search: all of them, or each asked area with everything under it (academic -> academic.math ...),
    minus the excluded ones and everything under them."""
    return [a for a in areas() if (not asked or under(a, asked)) and not under(a, exclude or [])]


class Doc(BaseModel):
    id: str
    text: str
    source: str = ""


class Upsert(BaseModel):
    area: str
    documents: list[Doc]


class Delete(BaseModel):
    area: str
    prefix: str | None = None
    prefixes: list[str] | None = None  # several at once: one save instead of one per prefix


class Search(BaseModel):
    query: str
    areas: list[str] | None = None
    exclude: list[str] | None = None
    limit: int = 5
    graph: int = 0  # GraphRAG: add up to this many chunks that the graph links to the hits
    min_weight: float = 0.75  # only follow edges at least this similar


@app.get("/health")
def health():
    return {"ok": True, "areas": areas()}


@app.get("/areas", dependencies=[Depends(auth)])
def list_areas():
    with _lock:
        return {a: index(a).count() for a in areas()}


@app.post("/upsert", dependencies=[Depends(auth)])
def upsert(req: Upsert):
    with _lock:
        emb = index(req.area, create=True)
        rows = [{"id": d.id, "text": d.text, "source": d.source} for d in req.documents]
        if emb.count():
            emb.upsert(rows)
        else:
            emb.index(rows)
        emb.save(str(DATA / req.area))
        return {"area": req.area, "count": emb.count()}


@app.post("/delete", dependencies=[Depends(auth)])
def delete(req: Delete):
    with _lock:
        emb = index(req.area)
        if emb is None or not emb.count():
            return {"area": req.area, "deleted": 0}
        ids = []
        for prefix in ([req.prefix] if req.prefix else []) + (req.prefixes or []):
            ids += ids_with_prefix(emb, prefix)
        if ids:
            emb.delete(ids)
            emb.save(str(DATA / req.area))
        return {"area": req.area, "deleted": len(ids)}


def neighbours(emb: Embeddings, ids: list[str], min_weight: float) -> list[tuple[str, float, str]]:
    """(chunk id, edge weight, hit it hangs on) for the graph neighbours of the given chunk ids."""
    g = emb.graph.backend if emb.graph is not None else None
    if g is None:
        return []
    node = {attrs.get("id"): n for n, attrs in g.nodes(data=True)}
    out = []
    for cid in ids:
        if cid not in node:
            continue
        for nb, d in g[node[cid]].items():
            w = d.get("weight", 0)
            if w >= min_weight:
                out.append((g.nodes[nb].get("id"), w, cid))
    return out


@app.post("/search", dependencies=[Depends(auth)])
def search(req: Search):
    hits = []
    with _lock:
        searched = []
        for area in expand(req.areas, req.exclude):
            emb = index(area)
            if emb is None or not emb.count():
                continue
            searched.append((area, emb))
            for r in emb.search("select id, text, source, score from txtai where similar(:q)",
                                limit=req.limit, parameters={"q": req.query}):
                hits.append({"area": area, **r})
        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = hits[: req.limit]
        if req.graph > 0:
            # GraphRAG: follow the graph from the hits; a neighbour scores hit score x edge weight
            have = {h["id"] for h in hits}
            score = {h["id"]: h["score"] for h in hits}
            extra = {}
            for area, emb in searched:
                for cid, w, via in neighbours(emb, [h["id"] for h in hits if h["area"] == area], req.min_weight):
                    if cid in have:
                        continue
                    s = score[via] * w
                    if s > extra.get(cid, (0,))[0]:
                        extra[cid] = (s, via, area, emb)
            for cid, (s, via, area, emb) in sorted(extra.items(), key=lambda x: -x[1][0])[: req.graph]:
                row = emb.search(f"select id, text, source from txtai where id = '{like(cid)}'", limit=1)
                if row:
                    hits.append({"area": area, **row[0], "score": s, "via": via})
    return hits


# ---- UI: graph view ----
def section_of(chunk_id: str) -> str:
    """'notes:docs/design.md#Setup#2' -> 'notes:docs/design.md#Setup'"""
    return chunk_id.rsplit("#", 1)[0]


@app.get("/ui/areas", dependencies=[Depends(ui_auth)])
def ui_areas():
    return list_areas()


@app.get("/ui/graph/{area}", dependencies=[Depends(ui_auth)])
def ui_graph(area: str):
    with _lock:
        emb = index(area)
        if emb is None or emb.graph is None:
            raise HTTPException(404, f"no area {area}")
        g = emb.graph.backend
        nodes, sec_of = {}, {}
        for n, attrs in g.nodes(data=True):
            sec = section_of(attrs.get("id", str(n)))
            sec_of[n] = sec
            repo, rest = sec.split(":", 1) if ":" in sec else ("", sec)
            file, _, name = rest.partition("#")
            node = nodes.setdefault(sec, {"id": sec, "label": name or file, "file": f"{repo}:{file}", "chunks": 0})
            node["chunks"] += 1
        edges = {}
        for u, v, d in g.edges(data=True):
            a, b = sorted((sec_of.get(u), sec_of.get(v)))
            if a and b and a != b:
                edges[(a, b)] = max(edges.get((a, b), 0), d.get("weight", 0))
    return {"nodes": list(nodes.values()),
            "edges": [{"source": a, "target": b, "weight": round(w, 3)} for (a, b), w in edges.items()]}


@app.get("/ui/section/{area}", dependencies=[Depends(ui_auth)])
def ui_section(area: str, key: str):
    with _lock:
        emb = index(area)
        if emb is None:
            raise HTTPException(404, f"no area {area}")
        ids = set(ids_with_prefix(emb, key + "#"))
        rows = [r for r in emb.search(f"select id, text, source from txtai where id like '{like(key + '#')}%'", limit=100000)
                if r["id"] in ids]
    rows.sort(key=lambda r: int(r["id"].rsplit("#", 1)[1]))
    return {"key": key, "source": rows[0]["source"] if rows else "", "text": "\n\n".join(r["text"] for r in rows)}


@app.get("/ui", response_class=HTMLResponse, dependencies=[Depends(ui_auth)])
def ui_page():
    return (Path(__file__).resolve().parent / "ui.html").read_text()
