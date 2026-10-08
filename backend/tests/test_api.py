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
PRIYA = {"X-Forwarded-For": "10.0.0.1", "X-Client-Id": "priya-browser-0001"}
RAHUL = {"X-Forwarded-For": "10.0.0.1", "X-Client-Id": "rahul-browser-0002"}  # same Wi-Fi


def make_client(**settings: object) -> tuple[TestClient, MemoryStore]:
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

    store = MemoryStore()
    app = create_app(Settings(_env_file=None, **settings), open_agent=opener, store=store)
    return TestClient(app), store


@pytest.fixture
def client() -> TestClient:
    return make_client()[0]


def upload(client: TestClient, headers=PRIYA, data: bytes | None = None):
    content = make_docx(SAMPLE_CV) if data is None else data
    return client.post("/runs", files={"cv": ("cv.docx", content)}, headers=headers)


def new_run(client: TestClient, headers=PRIYA) -> tuple[str, dict]:
    """Upload a CV; return the run id and the headers that unlock it."""
    body = upload(client, headers).json()
    return body["run_id"], {**headers, "X-Run-Token": body["run_token"]}


def resume(client, run_id, headers, body=RESUME, stream=False):
    return client.post(
        f"/runs/{run_id}/resume?stream={str(stream).lower()}", json=body, headers=headers
    )


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_and_presets(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert set(client.get("/presets").json()) == {"professional", "student"}


class TestStart:
    def test_upload_returns_the_profile_a_paused_run_and_its_key(self, client):
        response = upload(client)
        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "paused" and len(body["run_id"]) == 12
        assert len(body["run_token"]) >= 40
        assert body["profile"]["skills"][0]["name"] == "SQL"

    def test_unreadable_and_large_files(self, client):
        assert upload(client, data=b"not a cv").json()["detail"]["reason"] == "unsupported_type"
        assert upload(make_client(cv_max_bytes=1000)[0]).status_code == 413


class TestResume:
    def test_json_answer(self, client):
        run_id, unlocked = new_run(client)
        body = resume(client, run_id, unlocked).json()
        assert body["status"] == "finished" and body["jobs_found"] == 12
        assert len(body["matches"]) == 10

    def test_live_progress_then_the_result(self, client):
        run_id, unlocked = new_run(client)
        response = resume(client, run_id, unlocked, stream=True)
        events = sse_events(response.text)
        steps = [data["step"] for name, data in events if name == "progress"]
        assert steps[0] == "plan_query" and "score_jobs" in steps
        assert events[-1][0] == "result"

    def test_a_finished_run_can_be_viewed_but_not_resumed_again(self, client):
        run_id, unlocked = new_run(client)
        assert client.get(f"/runs/{run_id}", headers=unlocked).json()["status"] == "paused"
        resume(client, run_id, unlocked)
        assert client.get(f"/runs/{run_id}", headers=unlocked).json()["status"] == "finished"
        assert resume(client, run_id, unlocked).status_code == 409

    @pytest.mark.parametrize(
        ("changes", "message"),
        [({"country": "uae"}, "two-letter"), ({"preset": "wizard"}, "Unknown preset")],
    )
    def test_invalid_choices_are_explained(self, client, changes, message):
        run_id, unlocked = new_run(client)
        response = resume(client, run_id, unlocked, body={**RESUME, **changes})
        assert response.status_code == 422 and message in response.json()["detail"]


class TestRunKey:
    """Fix 1: a run belongs to whoever holds its key, not to an internet connection."""

    def test_the_owner_can_continue_after_changing_network(self, client):
        run_id, unlocked = new_run(client)
        on_mobile_data = {**unlocked, "X-Forwarded-For": "172.16.5.9"}
        assert resume(client, run_id, on_mobile_data).status_code == 200

    def test_others_cannot_open_the_run_even_on_the_same_wifi(self, client):
        run_id, _ = new_run(client, PRIYA)
        assert client.get(f"/runs/{run_id}", headers=RAHUL).status_code == 404
        guessed = {**RAHUL, "X-Run-Token": "a-guessed-key-of-the-right-length-000000000"}
        assert resume(client, run_id, guessed).status_code == 404


class TestDailyLimits:
    """Fix 2: limits count per browser; one network gets more room, but not unlimited."""

    def test_people_on_the_same_wifi_have_separate_limits(self):
        client, _ = make_client(api_starts_per_day=1)
        assert upload(client, PRIYA).status_code == 201
        assert upload(client, PRIYA).status_code == 429
        assert upload(client, RAHUL).status_code == 201  # same network, own limit

    def test_new_browser_ids_cannot_reset_the_limit_forever(self):
        client, _ = make_client(api_starts_per_day=1, api_ip_limit_multiplier=2)
        ids = [
            {"X-Forwarded-For": "10.9.9.9", "X-Client-Id": f"fresh-browser-id-{i:04d}"}
            for i in range(3)
        ]
        assert [upload(client, h).status_code for h in ids] == [201, 201, 429]

    def test_callers_without_a_browser_id_get_the_plain_limit(self):
        client, _ = make_client(api_starts_per_day=1)
        no_id = {"X-Forwarded-For": "10.7.7.7"}
        assert [upload(client, no_id).status_code for _ in range(2)] == [201, 429]

    def test_invite_code_required_when_configured(self):
        client, _ = make_client(api_invite_codes="alpha, beta")
        assert upload(client, {**PRIYA}).status_code == 401
        assert upload(client, {**PRIYA, "X-Invite-Code": "beta"}).status_code == 201


class TestBusy:
    """Fix 3: only N searches at once; the next person is told to try again shortly."""

    def test_busy_when_every_slot_is_taken_and_no_search_is_counted(self):
        client, store = make_client(api_max_concurrent_searches=1, api_resumes_per_day=1)
        run_id, unlocked = new_run(client)
        store.claim("jobscout:search-slot:0", "someone-else", ttl_seconds=60)
        busy = resume(client, run_id, unlocked)
        assert busy.status_code == 503 and busy.headers["retry-after"] == "60"
        store.release("jobscout:search-slot:0", "someone-else")
        assert resume(client, run_id, unlocked).status_code == 200  # the 503 cost nothing

    def test_the_slot_is_freed_after_each_search(self):
        client, store = make_client(api_max_concurrent_searches=1)
        for _ in range(2):
            run_id, unlocked = new_run(client)
            assert resume(client, run_id, unlocked, stream=True).status_code == 200
        assert store.get_json("jobscout:search-slot:0") is None


def test_cors_allows_the_website_and_its_headers(client):
    response = client.options(
        "/runs",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-client-id, x-run-token",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "x-run-token" in response.headers["access-control-allow-headers"].lower()
