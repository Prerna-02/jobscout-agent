# JobScout website

Next.js front end for the JobScout API: upload a CV, review the profile, choose the
scorecard, watch the search live, and read ranked results with evidence.

```bash
cp .env.example .env.local      # NEXT_PUBLIC_API_URL, default http://127.0.0.1:8000
npm install
npm run dev                     # http://localhost:3000 (start the API too: `uv run jobscout-api`)
```

| Script | What it does |
|---|---|
| `npm run types` | Regenerate `src/lib/api-types.ts` from the API's OpenAPI schema |
| `npm test` | Unit tests (stream parser, scorecard logic) |
| `npm run typecheck`, `npm run lint`, `npm run build` | The same checks CI runs |

The browser keeps an anonymous id (`X-Client-Id`, for fair daily limits) and each run's
secret key (`X-Run-Token`) in `localStorage`. The CV file is sent once and never stored.
