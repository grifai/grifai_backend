# grifai_backend

## Быстрый старт

```bash
bash scripts/dev.sh
```

- Запустит docker-compose (FastAPI + Postgres)
- Сгенерирует и применит миграции Alembic (если нужно)
- Откроет API на http://localhost:8080

### Остановка
```bash
docker-compose down
```

### Полезные команды
- Войти в контейнер API: `docker-compose exec api bash`
- Применить миграции вручную: `docker-compose exec api alembic upgrade head`
- Перегенерировать миграции: `docker-compose exec api alembic revision --autogenerate -m "msg"`
