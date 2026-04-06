#!/usr/bin/env bash
set -euo pipefail

DUMP_FILE="el_ripley_20260406.dump"
ENV_FILE=".env"
CONTAINER_NAME="el_ripley_postgres"

# ---------- Load .env ----------
if [[ ! -f "$ENV_FILE" ]]; then
  echo "[ERROR] $ENV_FILE not found. Run this script from the project root."
  exit 1
fi
set -a; source "$ENV_FILE"; set +a

# ---------- Validate dump ----------
if [[ ! -f "$DUMP_FILE" ]]; then
  echo "[ERROR] Dump file '$DUMP_FILE' not found."
  exit 1
fi

# ---------- Start containers ----------
echo "[1/4] Starting containers..."
docker compose up -d

echo "[2/4] Waiting for PostgreSQL to be healthy..."
until docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "healthy"; do
  printf "."
  sleep 2
done
echo " OK"

# ---------- Copy dump into container ----------
echo "[3/4] Copying dump file into container..."
docker cp "$DUMP_FILE" "$CONTAINER_NAME:/tmp/restore.dump"

# ---------- Restore ----------
echo "[4/4] Restoring database '$POSTGRES_DB'..."
docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" "$CONTAINER_NAME" \
  pg_restore \
    --username="$POSTGRES_USER" \
    --dbname="$POSTGRES_DB" \
    --no-owner \
    --no-privileges \
    --verbose \
    /tmp/restore.dump

echo ""
echo "Restore complete!"
echo "  Host     : localhost:${POSTGRES_PORT:-5434}"
echo "  Database : $POSTGRES_DB"
echo "  User     : $POSTGRES_USER"
echo "  Password : $POSTGRES_PASSWORD"
echo ""
echo "pgAdmin UI : http://localhost:${PGADMIN_PORT:-5050}"
echo "  Email    : $PGADMIN_DEFAULT_EMAIL"
echo "  Password : $PGADMIN_DEFAULT_PASSWORD"
