# Docker Deployment

## Services

- `api`: FastAPI backend.
- `worker`: optional background worker for Blender exports.
- `frontend`: React web player.
- `postgres`: dictionary metadata.
- `object-storage`: S3-compatible storage for JSON and exports.

## Commands

```powershell
docker compose up --build
```

The included compose file is development-oriented. Production should run Blender exports on isolated workers with CPU/GPU resource limits.
