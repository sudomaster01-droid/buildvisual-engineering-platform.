# Docker deployment

The Compose deployment persists SQLite data and generated media in the named
`buildvisual-data` volume. HTTPS termination is expected at a trusted reverse
proxy; production cookies are Secure and will not work over plain HTTP.

## First start

1. Install Docker Engine with the Compose plugin on the host.
2. Copy `backend/.env.example` to `backend/.env`.
3. Edit `backend/.env` and set:
   - `APP_ENV=production`
   - `APP_SECRET` to a unique random value of at least 32 characters
   - `PUBLIC_BASE_URL` to the final HTTPS address
   - `ENGINEER_PASSWORD` to a unique password of at least 10 characters
4. From the repository root, run:

   ```sh
   docker compose up --build -d
   ```

5. Check readiness:

   ```sh
   docker compose ps
   curl --fail http://localhost:8000/readyz
   ```

6. Configure the reverse proxy to route HTTPS traffic to port 8000. Keep the
   database/media volume and a backup of it when replacing the container.

Set `VIDEO_PROVIDER=local_render` if you want the truthful deterministic model
renderer. It does not generate AI footage. No production self-hosted video
checkpoint is currently shipped or verified. The optional Faster-Whisper
adapter requires the package and checkpoint described in
[OPEN_SOURCE_INVENTORY.md](./OPEN_SOURCE_INVENTORY.md); neither is downloaded by
the default container. To install the optional Faster-Whisper runtime, set
`INSTALL_LOCAL_AI=true` in the shell before building (or in the root Compose
`.env` file). Then place verified local weights in
`backend/models/faster-whisper-small` on the host to expose them read-only at
`/app/models`.

For local drawing interpretation, run a separately managed self-hosted
OpenAI-compatible vision endpoint, then set `LLM_PROVIDER=local_openai`,
`LLM_MODEL=Qwen/Qwen3-VL-2B-Instruct`, and `LLM_BASE_URL` to its reachable
`/v1` base URL. The application adapter is contract-tested; this repository
does not include the endpoint server or the 4.25 GB model weights.

Never commit `backend/.env`. Use an authenticated secret manager or a protected
host file with restrictive permissions for production credentials.
