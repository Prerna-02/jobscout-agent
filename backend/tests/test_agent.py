from datetime import UTC, date, datetime

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from jobscout.agent import AgentDeps, JobScoutAgent, RunNotFound, SearchPreferences
from jobscout.agent._planner import PlannedSearch, QueryPlanner
from jobscout.jobs import JobSearch, SearchQuery
from jobscout.profile import ProfileBuilder, ProfileExtraction
from jobscout.profile._extraction import ExtractedRole
from jobscout.storage import MemoryStore

from .conftest import SAMPLE_CV, make_docx
from .test_jobs import FakeJSearch

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
PREFS = SearchPreferences(
    target_titles=["Data Analyst"], country="ae", city="Dubai", skills=["SQL", "Python"]
)


def raw_job(i: int) -> dict:
    return {
        "job_id": f"id{i}",
        "job_title": f"Analyst role {i}",
        "employer_name": f"Company {i}",
        "job_apply_link": f"https://jobs.test/{i}",
        "job_country": "AE",
        "job_is_remote": False,
        "job_posted_at": "2 days ago",
        "job_description": "Analyse data with SQL.",
    }


def page(*ids: int) -> httpx.Response:
    return httpx.Response(200, json={"status": "OK", "data": {"jobs": [raw_job(i) for i in ids]}})


class FakePlannerLLM:
    """Writes a first search, and a broader one when asked to rewrite. Records each call."""

    def __init__(self, broader: str = "Data Analyst OR BI Analyst in Dubai"):
        self.calls: list[str] = []
        self.broader = broader
        self.runnable = RunnableLambda(self._answer)

    def _answer(self, messages) -> PlannedSearch:
        prompt = messages[-1].content
        self.calls.append(prompt)
        rewrite = "Too few jobs" in prompt
        return PlannedSearch(text=self.broader if rewrite else "Data Analyst SQL in Dubai")


EXTRACTION = ProfileExtraction(
    current_title="Data Analyst",
    skills=["SQL", "Python", "Tableau"],
    roles=[ExtractedRole(title="Data Analyst", employer="Acme", start="Jan 2022", end="Present")],
    locations=["Bengaluru, India"],
)


def make_agent(saver, fake_jsearch: FakeJSearch, planner_llm: FakePlannerLLM) -> JobScoutAgent:
    def deps() -> AgentDeps:
        return AgentDeps(
            profile_builder=ProfileBuilder(
                RunnableLambda(lambda _: EXTRACTION), today=date(2026, 10, 6)
            ),
            planner=QueryPlanner(planner_llm.runnable),
            job_search=lambda: JobSearch(fake_jsearch.client(), MemoryStore(), clock=lambda: NOW),
            clock=lambda: NOW,
        )

    return JobScoutAgent(saver, deps)


@pytest.fixture
def cv_bytes() -> bytes:
    return make_docx(SAMPLE_CV)


class TestPauseAndResume:
    def test_start_builds_the_profile_and_pauses(self, cv_bytes):
        agent = make_agent(InMemorySaver(), FakeJSearch(page(1)), FakePlannerLLM())
        paused = agent.start(cv_bytes)

        assert len(paused.run_id) == 12
        assert paused.profile.confirmed_skills == ["SQL", "Python"]
        assert paused.profile.flagged_skills == ["Tableau"]
        assert paused.suggested["target_titles"] == ["Data Analyst"]
        assert paused.suggested["city"] == "Bengaluru"
        assert paused.suggested["skills"] == ["SQL", "Python"]  # flagged skills not suggested

    def test_a_new_agent_object_can_resume_the_saved_run(self, cv_bytes):
        saver, fake = InMemorySaver(), FakeJSearch(page(*range(1, 13)))
        run_id = make_agent(saver, fake, FakePlannerLLM()).start(cv_bytes).run_id

        result = make_agent(saver, fake, FakePlannerLLM()).resume(run_id, PREFS)
        assert result.outcome == "enough_jobs"
        assert len(result.jobs) == 12

    def test_unknown_or_finished_runs_cannot_be_resumed(self, cv_bytes):
        agent = make_agent(InMemorySaver(), FakeJSearch(page(*range(1, 13))), FakePlannerLLM())
        with pytest.raises(RunNotFound):
            agent.resume("nope", PREFS)
        run_id = agent.start(cv_bytes).run_id
        agent.resume(run_id, PREFS)
        with pytest.raises(RunNotFound):
            agent.resume(run_id, PREFS)


class TestSearchLoop:
    def run(self, cv_bytes, fake: FakeJSearch, llm: FakePlannerLLM | None = None):
        llm = llm or FakePlannerLLM()
        agent = make_agent(InMemorySaver(), fake, llm)
        return agent.resume(agent.start(cv_bytes).run_id, PREFS), llm

    def test_broadens_until_enough_jobs(self, cv_bytes):
        fake = FakeJSearch(page(1, 2, 3, 4), page(5, 6, 7), page(8, 9, 10, 11, 12))
        result, llm = self.run(cv_bytes, fake)

        assert result.outcome == "enough_jobs"
        assert [s["rung"] for s in result.searches] == ["first", "similar_titles", "wider_dates"]
        assert [s["kept"] for s in result.searches] == [4, 3, 5]
        assert result.searches[1]["text"] == "Data Analyst OR BI Analyst in Dubai"
        assert result.searches[2]["text"] == result.searches[1]["text"]  # same words...
        assert result.searches[2]["date_posted"] == "month"  # ...older postings allowed
        assert len(llm.calls) == 2  # widening dates needs no LLM call
        assert len(result.jobs) == 12

    def test_stops_after_three_rewrites_and_says_so(self, cv_bytes):
        fake = FakeJSearch(page(1, 2), page(3, 4), page(5, 6), page(7, 8))
        result, _ = self.run(cv_bytes, fake)

        assert result.outcome == "retries_used"
        assert len(result.searches) == 4
        assert result.searches[-1] == {
            "text": "Data Analyst OR BI Analyst",  # city dropped, alternative title kept
            "date_posted": "month",
            "rung": "wider_area",
            "found": 2,
            "cache_hit": False,
            "kept": 2,
        }
        assert any("Only 8 matching jobs" in w for w in result.warnings)

    def test_a_rewrite_that_repeats_a_search_ends_the_loop(self, cv_bytes):
        fake = FakeJSearch(page(1, 2))
        result, _ = self.run(cv_bytes, fake, FakePlannerLLM(broader="Data Analyst SQL in Dubai"))

        assert result.outcome == "no_new_search"
        assert len(result.searches) == 1
        assert len(fake.requests) == 1

    def test_the_same_ad_from_two_searches_is_kept_once(self, cv_bytes):
        fake = FakeJSearch(page(1, 2, 3, 4), page(3, 4, 5, 6, 7, 8, 9, 10))
        result, _ = self.run(cv_bytes, fake)

        assert [s["kept"] for s in result.searches] == [4, 6]
        assert result.dropped == {"duplicate": 2}
        assert len(result.jobs) == 10

    def test_search_errors_end_the_run_with_a_clear_message(self, cv_bytes):
        result, _ = self.run(cv_bytes, FakeJSearch(httpx.Response(401)))

        assert result.outcome == "search_failed"
        assert result.jobs == []
        assert any("API key" in w for w in result.warnings)


class TestQueryPlanner:
    def test_falls_back_to_a_template_when_the_llm_fails(self):
        def broken(_):
            raise RuntimeError("model down")

        query = QueryPlanner(RunnableLambda(broken)).first(PREFS)
        assert query.text == "Data Analyst in Dubai"
        assert (query.country, query.date_posted) == ("ae", "week")

    def test_rejects_llm_text_with_personal_details(self):
        llm = RunnableLambda(lambda _: PlannedSearch(text="analyst jane@example.com"))
        assert QueryPlanner(llm).first(PREFS).text == "Data Analyst in Dubai"

    def test_country_and_dates_come_from_preferences_not_the_llm(self):
        llm = RunnableLambda(lambda _: PlannedSearch(text="data analyst in London"))
        query = QueryPlanner(llm).first(PREFS)
        assert query.country == "ae"  # the model cannot switch countries

    def test_wider_area_drops_the_city(self):
        previous = SearchQuery(text="Data Analyst in Dubai", country="ae", date_posted="month")
        query = QueryPlanner(RunnableLambda(lambda _: None)).broaden(previous, PREFS, "wider_area")
        assert query.text == "Data Analyst"
        assert query.date_posted == "month"


class TestPreferences:
    @pytest.mark.parametrize(
        "changes",
        [
            {"country": "uae"},
            {"target_titles": []},
            {"target_titles": ["Analyst {{ignore rules}}"]},
            {"city": "<script>"},
            {"skills": ["x" * 61]},
        ],
    )
    def test_invalid_answers_are_rejected(self, changes):
        data = {**PREFS.model_dump(), **changes}
        with pytest.raises(ValidationError):
            SearchPreferences(**data)


class TestScoringInTheGraph:
    def test_resume_returns_ranked_matches_with_the_chosen_preset(self, cv_bytes):
        from jobscout.scoring import PRESETS, MatchScorer, Shortlister
        from jobscout.scoring._scorer import BatchAssessment, CriterionScore, JobAssessment
        from jobscout.storage import MemoryStore

        from .test_scoring import KeywordEmbeddings

        def score_by_number(messages):
            # Higher job number -> higher skills score, so the ranking is predictable.
            ids = [
                ln.split('"')[1]
                for ln in messages[1].content.splitlines()
                if ln.startswith("<job id=")
            ]
            return BatchAssessment(
                jobs=[
                    JobAssessment(
                        job_id=i,
                        scores=[
                            CriterionScore(
                                criterion="skills",
                                score=int(i[1:]) % 11,
                                job_quote="Analyse data with SQL",
                            )
                        ],
                    )
                    for i in ids
                ]
            )

        fake = FakeJSearch(page(*range(1, 13)))
        base = make_agent(InMemorySaver(), fake, FakePlannerLLM())._deps

        def deps():
            d = base()
            d.shortlister = lambda: Shortlister(
                KeywordEmbeddings(), MemoryStore(), model_name="m", size=8
            )
            d.scorer = lambda: MatchScorer(RunnableLambda(score_by_number))
            d.top_n = 3
            return d

        agent = JobScoutAgent(InMemorySaver(), deps)
        prefs = PREFS.model_copy(update={"rubric": PRESETS["student"]})
        result = agent.resume(agent.start(cv_bytes).run_id, prefs)

        assert result.rubric_name == "Student"
        assert len(result.matches) == 3
        totals = [m.total for m in result.matches]
        assert totals == sorted(totals, reverse=True)
        assert all(m.criteria[0].key == "skills" for m in result.matches)
        _, _, rubric, pairs = agent.scoring_inputs(result.run_id)
        assert rubric.name == "Student" and len(pairs) == 8


def test_scam_postings_are_removed_before_scoring(cv_bytes):
    from jobscout.scoring import MatchScorer, Shortlister
    from jobscout.storage import MemoryStore
    from jobscout.trust import TrustChecker

    from .test_scoring import FakeScorerLLM, KeywordEmbeddings
    from .test_trust import FakeReviewer

    jobs = [raw_job(i) for i in range(1, 12)]
    jobs[0]["job_description"] = "Great role. A registration fee of AED 500 is required to start."
    fake = FakeJSearch(httpx.Response(200, json={"status": "OK", "data": {"jobs": jobs}}))
    scorer_llm = FakeScorerLLM()
    base = make_agent(InMemorySaver(), fake, FakePlannerLLM())._deps

    def deps():
        d = base()
        d.trust = lambda: TrustChecker(FakeReviewer(verdict="no_concerns", quote=None).runnable)
        d.shortlister = lambda: Shortlister(KeywordEmbeddings(), MemoryStore(), model_name="m")
        d.scorer = lambda: MatchScorer(scorer_llm.runnable)
        return d

    agent = JobScoutAgent(InMemorySaver(), deps)
    result = agent.resume(agent.start(cv_bytes).run_id, PREFS)

    assert result.removed_as_risky == [
        {
            "title": "Analyst role 1",
            "company": "Company 1",
            "reason": "The ad asks candidates to pay money",
        }
    ]
    assert result.dropped["scam_risk"] == 1
    scored_text = " ".join(call[1].content for call in scorer_llm.calls)
    assert "registration fee" not in scored_text  # Claude never spent tokens on the scam
    assert all(m.job.job_id != "id1" for m in result.matches)
