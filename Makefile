# Build docker image
build:
	docker compose build

# Jalankan API (default CMD)
up:
	docker compose up -d

# Hentikan container
down:
	docker compose down

# Jalankan PDF ingestion (sekali saja, ganti sesuai nama file)
ingest:
	docker compose run --rm chatbox python app.py --pdf ./docs/BUMN.pdf

# Lihat log container
logs:
	docker compose logs -f

# Rebuild dan restart
reup: down build up
