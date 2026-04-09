# Hermes Ladybug Memory

Ladybug Memory as a standalone Hermes memory-provider plugin.

It gives Hermes a fully local, file-based memory store backed by [LadybugMemory](https://github.com/Ladybug-Memory/ladybug-memory/) — a columnar embedded graph database (`.lbdb`). No API keys, no cloud, no sync service. Everything stays on disk in `HERMES_HOME`.

## Why Ladybug Memory

Ladybug Memory is a local-first memory backend for Hermes that keeps all data in a single `.lbdb` file on your machine. It supports BM25 keyword search, importance-weighted recall, typed memory entries, named graph edges between entries, and optional GLiNER2 entity extraction — all without any external service.

### Where It Fits

- Best fit if you want local, inspectable memory with structured recall and graph relationships, and no SaaS dependency.
- Compared with Honcho, Mem0, and RetainDB, this is fully local with no hosted component.
- Compared with Observational Memory and Holographic, this is less about shared cross-agent markdown stores and more about a typed, searchable, graph-linked memory store owned entirely by Hermes.
- Compared with OpenViking and ByteRover, this is simpler and more direct — no hierarchical browser or knowledge-graph UI, just a fast embedded database with the tools the agent needs.

## Install

Install the plugin and its Python dependency:

```bash
hermes plugins install Ladybug-Memory/hermes-memory-plugin
pip install ladybug-memory
```

Then link the plugin into the memory provider directory so `hermes memory setup` can discover it:

```bash
ln -s ~/.hermes/plugins/ladybug \
      ~/.hermes/hermes-agent/plugins/memory/ladybug
```

> **Why the symlink?** Hermes's memory provider system currently only discovers providers bundled in `plugins/memory/` inside the hermes-agent source tree. User-installed plugins (`~/.hermes/plugins/`) are not scanned by the memory discovery system yet. This symlink bridges the gap. See [NousResearch/hermes-agent#4956](https://github.com/NousResearch/hermes-agent/issues/4956) for the upstream feature request.

Finally, configure it:

```bash
hermes memory setup    # select "ladybug"
```

## Requirements

- Hermes with the memory-provider plugin system
- `ladybug-memory` >= 0.1.4

Install into the Hermes runtime environment if you do not already have it:

```bash
pip install ladybug-memory
```

For GLiNER2 entity extraction (optional):

```bash
pip install ladybug-memory[extract]
```

## Manual Install

If you prefer cloning manually:

```bash
git clone https://github.com/Ladybug-Memory/hermes-memory-plugin.git \
  ~/.hermes/plugins/ladybug
ln -s ~/.hermes/plugins/ladybug \
      ~/.hermes/hermes-agent/plugins/memory/ladybug
pip install ladybug-memory
hermes memory setup
```

## What It Adds

**Tools:**
- `ladybug_store`: persist a new memory entry with type and importance score
- `ladybug_search`: BM25 keyword search across stored memories
- `ladybug_recall`: retrieve recent or high-importance memories
- `ladybug_update`: correct or update a memory by ID
- `ladybug_delete`: delete a memory by ID
- `ladybug_link`: create a named relationship between two memories
- `ladybug_related`: traverse the memory graph by relationship
- `ladybug_entity`: entity-level KG queries via GLiNER2 (optional)

**Memory integration:**
- background prefetch before every turn (importance-weighted recall + query search)
- mirrors built-in `MEMORY.md` / `USER.md` writes into Ladybug automatically
- surfaces high-importance memories during context compression

## Config

All keys go under `memory.ladybug` in `~/.hermes/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/ladybug.lbdb` | Path to the Ladybug database file |
| `prefetch_limit` | `6` | Memories surfaced before each turn |
| `min_importance` | `3` | Minimum importance score for prefetch recall |
| `auto_link` | `false` | Auto-link mirrored built-in memory writes |

## Memory Types

`general` · `preference` · `fact` · `project` · `person` · `event` · `task`

## Importance Scores

1–10 scale. Higher scores surface more often in prefetch recall. Built-in `MEMORY.md` / `USER.md` mirrors use importance **6** (explicit user signal). Tune with `ladybug_update` over time.

## Validation

This repository ships standalone tests for the provider behavior. Run them with:

```bash
uv run --with pytest pytest tests -q
```

## Notes

- This repository is laid out as a Hermes directory plugin, so the repo root is the plugin root.
- The installed plugin name is `ladybug`, matching the GitHub repo's directory name.
- Hermes currently clones directory plugins from Git but does not install their Python dependencies automatically, so the `pip install ladybug-memory` step is still required.
