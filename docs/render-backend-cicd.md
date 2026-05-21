# Render Backend CI/CD

This backend deploys to Render with GitHub Actions and a Render Deploy Hook.

## Render Web Service

Create a Render Web Service from this backend repository.

- Runtime: Python
- Branch: `main`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`

Recommended Render deploy setting:

- Disable regular auto deploys, then let GitHub Actions trigger deploys with the deploy hook after checks pass.
- Alternatively, use Render's "After CI checks pass" auto-deploy mode and remove the deploy hook step from the workflow.

## Render Environment Variables

Set production values in Render Dashboard. Do not commit `.env`.

- `DATABASE_URL`
- `POSTGRES_SSLMODE=require`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `JWT_SECRET_KEY`
- `ITEMS_FEATURE_PATH=data/Items_Feature.csv`
- `USER_PREFERENCE_VECTOR_PATH=data/recommendation_output/final/user_preference_vectors_expanded.csv`
- `USER_PREFERENCE_CLUSTER_PATH=data/recommendation_output/final/user_preference_clusters.csv`
- `USER_PREFERENCE_CENTROID_PATH=data/recommendation_output/final/user_preference_cluster_centroids.csv`

## GitHub Repository Secret

In GitHub:

1. Go to `Settings > Secrets and variables > Actions`.
2. Add `RENDER_DEPLOY_HOOK_URL`.
3. Paste the Deploy Hook URL from the Render service settings.

## Workflow

- Pull request to `main`: install dependencies, compile Python files, run backend smoke checks.
- Push or merge to `main`: run the same checks, then trigger Render deploy.

The smoke check verifies:

- The 9 main API routes are registered.
- Recommendation data files exist.
- User-based recommendation returns exactly one candidate.
- Cluster-based recommendation returns all deduplicated candidates in the matched group.
