.PHONY: up down logs

# One-command start: fetch LFS seed dump then bring up the stack.
up:
	git lfs pull
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f mcp-docs
