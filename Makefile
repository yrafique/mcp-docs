.PHONY: up down clean logs

# Single self-contained container. Run these from the repo root.
COMPOSE = docker compose -f docker/compose.yml

# Build + start the container (defaults: nsp-26.4 + sros-26.3).
up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

# Clean rebuild: wipe the built schemas so they re-import from the dumps.
clean:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f mcp-docs
