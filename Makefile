.PHONY: dev prod migrate index index-update index-force ask summary stats logs test stop clean install lint

# ── Development ───────────────────────────────────────────────────────────────

dev:
	docker-compose up -d postgres redis qdrant
	python -m app.main

prod:
	docker-compose up -d

# ── Data / indexing ───────────────────────────────────────────────────────────

migrate:
	python scripts/migrate_memory.py
	python scripts/migrate_rag.py

index:
	python scripts/index.py

index-force:
	python scripts/index.py --force

index-update:
	python scripts/index.py --update

# ── Queries ───────────────────────────────────────────────────────────────────

ask:
	@read -p "Query: " q; python -m app.services.search "$$q"

summary:
	python -m app.services.digest

stats:
	python scripts/stats.py

# ── Docker helpers ────────────────────────────────────────────────────────────

logs:
	docker-compose logs -f jarvis

stop:
	docker-compose down

clean:
	docker-compose down -v

# ── Local dev helpers ─────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	python -m py_compile \
		app/main.py \
		app/config.py \
		app/bot/client.py \
		app/llm/openai_provider.py \
		app/memory/rag.py \
		app/memory/contacts.py \
		app/services/search.py \
		app/services/digest.py \
		app/services/ghost_writer.py \
		app/services/tts.py \
		app/services/event_bus.py \
		app/services/consumers.py \
		app/api/server.py \
		scripts/index.py
	@echo "All files compile OK"
