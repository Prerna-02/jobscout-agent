import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import fakeredis
import httpx
import pytest
import redis
from pydantic import ValidationError

from jobscout.jobs import (
    FilterRules,
    JobPosting,
    JobSearch,
    JSearchError,
    SearchQuery,
    filter_jobs,
)
from jobscout.jobs._client import JSearchClient
from jobscout.jobs._normalize import normalize_job, parse_relative_age
from jobscout.jobs._store import MemoryStore, RedisStore

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "jsearch_dubai_data_analyst.json").read_text("utf-8")
)
RAW_JOBS = FIXTURE["data"]["jobs"]


class FakeJSearch:
    """An httpx transport that replays queued responses and records every request."""

    def __init__(self, *responses: httpx.Response | Exception):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    def client(self) -> JSearchClient:
        return JSearchClient(
            "test-key",
            base_url="https://api.test/jsearch",
            transport=httpx.MockTransport(self),
            sleep=lambda _: None,
        )


def ok(body=FIXTURE) -> httpx.Response:
    return httpx.Response(200, json=body)


class TestSearchQuery:
    def test_params_always_ask_for_english(self):
        q = SearchQuery(
            text="  data   analyst in Dubai ",
            country="AE",
            remote_only=True,
            employment_types=("INTERN", "FULLTIME"),
        )
        assert q.to_params() == {
            "query": "data analyst in Dubai",
            "country": "ae",
            "language": "en",
            "date_posted": "week",
            "work_from_home": "true",
            "employment_types": "FULLTIME,INTERN",
        }

    def test_same_search_same_cache_key(self):
        a = SearchQuery(text="Data Analyst", employment_types=("INTERN", "FULLTIME"))
        b = SearchQuery(text="data analyst", employment_types=("FULLTIME", "INTERN"))
        assert a.cache_key() == b.cache_key()
        assert a.cache_key() != SearchQuery(text="data analyst", country="ae").cache_key()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("text", "analyst jane@example.com"),
            ("text", "analyst +91 98765 43210"),
            ("text", "[NAME] analyst"),
            ("text", "analyst {{system}}"),
            ("text", "x" * 101),
            ("country", "uae"),
            ("date_posted", "yesterday"),
        ],
    )
    def test_rejects_unsafe_or_invalid_input(self, field, value):
        args = {"text": "data analyst", field: value}
        with pytest.raises(ValidationError):
            SearchQuery(**args)


class TestNormalize:
    def test_real_response_is_normalized(self):
        jobs = [normalize_job(raw, now=NOW) for raw in RAW_JOBS]
        assert all(jobs) and len(jobs) == 10
        first = jobs[0]
        assert first.title.startswith("Data Insights Analyst")
        assert first.company == "Avanta Works"
        assert first.country == "AE"
        assert first.apply_link.startswith("https://")

    def test_jobs_without_an_apply_link_are_dropped(self):
        assert normalize_job({**RAW_JOBS[0], "job_apply_link": None}, now=NOW) is None

    @pytest.mark.parametrize(
        ("text", "age"),
        [
            ("3 days ago", timedelta(days=3)),
            ("an hour ago", timedelta(hours=1)),
            ("30+ days ago", timedelta(days=30)),
            ("2 weeks ago", timedelta(weeks=2)),
            ("Just posted", timedelta(0)),
            ("yesterday", timedelta(days=1)),
        ],
    )
    def test_relative_posting_dates(self, text, age):
        assert parse_relative_age(text, now=NOW) == NOW - age

    def test_unknown_date_text_is_not_guessed(self):
        assert parse_relative_age("قبل 3 أيام", now=NOW) is None

    def test_salary_from_numbers(self):
        raw = {**RAW_JOBS[0], "job_min_salary": 8000, "job_max_salary": 12000}
        raw["job_salary_period"] = "MONTH"
        assert normalize_job(raw, now=NOW).salary == "8,000-12,000 per month"


class TestClient:
    def test_sends_key_and_params_to_search_v2(self):
        fake = FakeJSearch(ok())
        jobs = fake.client().search({"query": "data analyst", "language": "en"})
        request = fake.requests[0]
        assert len(jobs) == 10
        assert request.url.path == "/jsearch/search-v2"
        assert request.headers["x-api-key"] == "test-key"
        assert request.url.params["language"] == "en"

    def test_retries_server_errors_and_timeouts(self):
        fake = FakeJSearch(httpx.Response(503), httpx.ConnectTimeout("slow"), ok())
        assert len(fake.client().search({})) == 10
        assert len(fake.requests) == 3

    def test_gives_up_after_three_attempts(self):
        fake = FakeJSearch(httpx.Response(502))
        with pytest.raises(JSearchError, match="having problems"):
            fake.client().search({})
        assert len(fake.requests) == 3

    @pytest.mark.parametrize(("status", "message"), [(401, "API key"), (429, "limit")])
    def test_does_not_retry_key_or_quota_errors(self, status, message):
        fake = FakeJSearch(httpx.Response(status))
        with pytest.raises(JSearchError, match=message):
            fake.client().search({})
        assert len(fake.requests) == 1

    def test_unexpected_body_is_an_error(self):
        fake = FakeJSearch(httpx.Response(200, json={"status": "ERROR", "data": {"jobs": []}}))
        with pytest.raises(JSearchError, match="error instead of results"):
            fake.client().search({})


class TestJobSearch:
    QUERY = SearchQuery(text="data analyst in Dubai", country="ae")

    def make(self, fake: FakeJSearch, **limits) -> JobSearch:
        return JobSearch(fake.client(), MemoryStore(), clock=lambda: NOW, **limits)

    def test_second_identical_search_comes_from_cache(self):
        fake = FakeJSearch(ok())
        search = self.make(fake)
        first, second = search.search(self.QUERY), search.search(self.QUERY)
        assert (first.cache_hit, second.cache_hit) == (False, True)
        assert second.jobs == first.jobs
        assert len(fake.requests) == 1 and search.calls_made == 1

    def test_empty_results_are_cached_too(self):
        fake = FakeJSearch(ok({"status": "OK", "data": {"jobs": []}}))
        search = self.make(fake)
        search.search(self.QUERY)
        assert search.search(self.QUERY).cache_hit
        assert len(fake.requests) == 1

    def test_per_run_budget_stops_a_runaway_loop(self):
        fake = FakeJSearch(ok())
        search = self.make(fake, max_calls_per_run=2)
        search.search(SearchQuery(text="analyst one"))
        search.search(SearchQuery(text="analyst two"))
        with pytest.raises(JSearchError, match="already used its 2"):
            search.search(SearchQuery(text="analyst three"))
        assert len(fake.requests) == 2

    def test_monthly_limit_is_shared_across_runs(self):
        store, fake = MemoryStore(), FakeJSearch(ok())
        for text in ("analyst one", "analyst two"):
            JobSearch(fake.client(), store, monthly_limit=2, clock=lambda: NOW).search(
                SearchQuery(text=text)
            )
        new_run = JobSearch(fake.client(), store, monthly_limit=2, clock=lambda: NOW)
        with pytest.raises(JSearchError, match="month's job search limit"):
            new_run.search(SearchQuery(text="analyst three"))
        assert new_run.search(SearchQuery(text="analyst one")).cache_hit  # cache still works


class TestRedisStore:
    def test_round_trip_and_counter(self):
        store = RedisStore(fakeredis.FakeRedis(decode_responses=True))
        store.set_json("k", {"jobs": [1, 2]}, ttl_seconds=60)
        assert store.get_json("k") == {"jobs": [1, 2]}
        assert [store.incr("c", ttl_seconds=60) for _ in range(3)] == [1, 2, 3]

    def test_redis_outage_never_breaks_a_search(self):
        class Down:
            def __getattr__(self, name):
                def fail(*args, **kwargs):
                    raise redis.ConnectionError("down")

                return fail

        store = RedisStore(Down())  # type: ignore[arg-type]
        assert store.get_json("k") is None
        store.set_json("k", {}, ttl_seconds=60)
        assert store.incr("c", ttl_seconds=60) is None

    @pytest.mark.parametrize(
        "url", ["redis-cli --tls -u redis://default:x@host:6379", "host:6379", ""]
    )
    def test_rejects_urls_that_are_not_redis_urls(self, url):
        with pytest.raises(ValueError, match="rediss://"):
            RedisStore.from_url(url)

    def test_accepts_quoted_tls_url(self):
        assert RedisStore.from_url('"rediss://default:x@example.upstash.io:6379"')


def job(job_id: str, title: str = "Analyst", company: str = "Acme", **extra) -> JobPosting:
    return JobPosting(job_id=job_id, title=title, company=company, apply_link="https://x", **extra)


class TestFilter:
    def test_duplicates_by_id_and_by_syndicated_ad(self):
        jobs = [job("1"), job("1"), job("2", "ANALYST", "acme"), job("3", "Engineer")]
        result = filter_jobs(jobs, FilterRules(), now=NOW)
        assert [j.job_id for j in result.kept] == ["1", "3"]
        assert result.dropped == {"duplicate": 2}

    def test_rules(self):
        jobs = [
            job("remote", "A", is_remote=True, country="US"),
            job("wrong", "B", country="US"),
            job("old", "C", country="AE", posted_at=NOW - timedelta(days=40)),
            job("undated", "D", country="AE"),
            job("seen", "E", country="AE"),
        ]
        rules = FilterRules(
            max_age_days=30, countries=frozenset({"ae"}), exclude_ids=frozenset({"seen"})
        )
        result = filter_jobs(jobs, rules, now=NOW)
        assert [j.job_id for j in result.kept] == ["remote", "undated"]
        assert result.dropped == {"wrong_country": 1, "too_old": 1, "duplicate": 1}

    def test_remote_only(self):
        result = filter_jobs(
            [job("1", is_remote=True), job("2", "B")], FilterRules(remote_only=True), now=NOW
        )
        assert [j.job_id for j in result.kept] == ["1"]
