#!/bin/bash
set -e

# Запуск docker-compose
if ! docker-compose ps | grep -q 'api'; then
  echo "[dev] Запуск docker-compose..."
  docker-compose up -d
else
  echo "[dev] docker-compose уже запущен."
fi

# Ожидание готовности Postgres
until docker-compose exec db pg_isready -U postgres > /dev/null 2>&1; do
  echo "[dev] Ожидание Postgres..."
  sleep 1
done

# Генерация и применение миграций Alembic
if docker-compose exec api alembic current | grep -q 'head'; then
  echo "[dev] Alembic: миграции уже применены."
else
  echo "[dev] Alembic: автогенерация и применение миграции..."
  docker-compose exec api alembic revision --autogenerate -m "init" || true
  docker-compose exec api alembic upgrade head
fi

# Вывод статуса
echo "[dev] Проект запущен: http://localhost:8080"
echo "[dev] Для остановки: docker-compose down"
