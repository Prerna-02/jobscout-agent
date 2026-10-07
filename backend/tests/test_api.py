import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from jobscout.agent import JobScoutAgent
from jobscout.api import create_app
from jobscout.scoring import MatchScorer, Shortlister
from jobscout.settings import Settings
from jobscout.storage import MemoryStore

from .conftest import SAMPLE_CV, make_docx
from .test_agent import FakeJSearch, FakePlannerLLM, make_agent, page
from .test_scoring import FakeScorerLLM, KeywordEmbeddings

RESUME = {"target_titles": ["Data Analyst"], "country": "ae", "city": "Dubai"}
USER_A = {"X-Forwarded-For": "10.0.0.1"}
USER_B = {"X-Forwarded-For": "10.0.0.2"}


def make_client(**settings: object) -> TestClient:
    base = make_agent(InMemorySaver(), FakeJSearch(page(*range(1, 13))), FakePlannerLLM())._deps
    scorer_llm = FakeScorerLLM()

    def deps():
        d = base()
        d.shortlister = lambda: Shortlister(KeywordEmbeddings(), MemoryStore(), model_name="m")
        d.scorer = lambda: MatchScorer(scorer_llm.runnable)
        return d

    agent = JobScoutAgent(InMemorySaver(), deps)

    @contextmanager
    def opener():
        yield agent

    app = create_app(Settings(_env_file=None, **settings), open_agent=opener, store=MemoryStore())
    return TestClient(app)


def upload(client: TestClient, headers=USER_A, data: bytes | None = None):
    content = make_docx(SAMPLE_CV) if data is None else data
    return client.post("/runs", files={"cv": ("cv.docx", content)}, headers=headers)


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client() -> TestClient:
    return make_client()


def test_health_and_presets(client):
    assert client.get("/health").json() == {"status": "ok"}
    presets = client.get("/presets").json()
    assert set(presets) == {"professional", "student"}
    assert sum(c["weight"] for c in presets["student"]["criteria"]) == 100


class TestStart:
    def test_upload_returns_the_profile_and_a_paused_run(self, client):
        response = upload(client)
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "paused" and len(body["run_id"]) == 12
        assert body["profile"]["skills"][0]["name"] == "SQL"
        assert body["suggested"]["preset"] == "professional"
        assert set(body["presets"]) == {"professional", "student"}

    def test_unreadable_files_get_a_reason(self, client):
        response = upload(client, data=b"not a cv")
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == "unsupported_type"

    def test_large_files_are_refused_before_parsing(self):
        response = upload(make_client(cv_max_bytes=1000))
        assert response.status_code == 413


class TestResume:
    def test_json_answer(self, client):
        run_id = upload(client).json()["run_id"]
        response = client.post(f"/runs/{run_id}/resume?stream=false", json=RESUME, headers=USER_A)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "finished" and body["outcome"] == "enough_jobs"
        assert body["jobs_found"] == 12 and len(body["matches"]) == 10
        # The fake scorer's quote isn't in these fake ads, so every score is discarded:
        assert (
            body["matches"][0]["criteria"][0]["note"] == "quote not found in the ad, score ignored"
        )

    def test_live_progress_then_the_result(self, client):
        run_id = upload(client).json()["run_id"]
        response = client.post(f"/runs/{run_id}/resume", json=RESUME, headers=USER_A)
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse_events(response.text)
        steps = [data["step"] for name, data in events if name == "progress"]
        assert steps[0] == "plan_query" and "search_jobs" in steps and "score_jobs" in steps
        assert events[-1][0] == "result"
        assert events[-1][1]["matches"]

    def test_a_finished_run_can_be_viewed_but_not_resumed_again(self, client):
        run_id = upload(client).json()["run_id"]
        assert client.get(f"/runs/{run_id}", headers=USER_A).json()["status"] == "paused"
        client.post(f"/runs/{run_id}/resume?stream=false", json=RESUME, headers=USER_A)
        assert client.get(f"/runs/{run_id}", headers=USER_A).json()["status"] == "finished"
        again = client.post(f"/runs/{run_id}/resume", json=RESUME, headers=USER_A)
        assert again.status_code == 409

    @pytest.mark.parametrize(
        ("changes", "message"),
        [({"country": "uae"}, "two-letter"), ({"preset": "wizard"}, "Unknown preset")],
    )
    def test_invalid_choices_are_explained(self, client, changes, message):
        run_id = upload(client).json()["run_id"]
        response = client.post(f"/runs/{run_id}/resume", json={**RESUME, **changes}, headers=USER_A)
        assert response.status_code == 422 and message in response.json()["detail"]


class TestGuard:
    def test_users_only_see_their_own_runs(self, client):
        run_id = upload(client, headers=USER_A).json()["run_id"]
        assert client.get(f"/runs/{run_id}", headers=USER_B).status_code == 404
        resume = client.post(f"/runs/{run_id}/resume", json=RESUME, headers=USER_B)
        assert resume.status_code == 404

    def test_invite_code_required_when_configured(self):
        client = make_client(api_invite_codes="alpha, beta")
        assert upload(client, headers={}).status_code == 401
        assert upload(client, headers={"X-Invite-Code": "wrong"}).status_code == 401
        assert upload(client, headers={"X-Invite-Code": "beta"}).status_code == 201

    def test_daily_limits(self):
        client = make_client(api_starts_per_day=1, api_resumes_per_day=1)
        run_id = upload(client).json()["run_id"]
        assert upload(client).status_code == 429
        client.post(f"/runs/{run_id}/resume?stream=false", json=RESUME, headers=USER_A)
        assert upload(client, headers=USER_B).status_code == 201  # other users are unaffected

    def test_cors_allows_the_website(self, client):
        response = client.options(
            "/runs",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
