# Nama layanan
SERVICE=chatbox
DOCKER_COMPOSE=docker-compose

.PHONY: build up down logs clean restart

# Membangun image Docker
build:
	$(DOCKER_COMPOSE) build

# Menjalankan seluruh layanan
up:
	$(DOCKER_COMPOSE) up -d

# Menghentikan semua container
down:
	$(DOCKER_COMPOSE) down

# Melihat log dari layanan chatbox (bisa diganti dengan ollama jika perlu)
logs:
	$(DOCKER_COMPOSE) logs -f $(SERVICE)

# Membersihkan volume dan cache
clean:
	$(DOCKER_COMPOSE) down -v
	rm -rf db/

# Restart service (down dan up ulang)
restart: down up

# Jalankan hanya service chatbox
run-chatbox:
	docker-compose run --rm $(SERVICE)

# Masuk ke shell dalam container chatbox
shell:
	docker exec -it chatbox /bin/bash

# Masuk ke shell dalam container ollama
shell-ollama:
	docker exec -it ollama /bin/bash
