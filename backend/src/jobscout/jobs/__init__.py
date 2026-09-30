"""Job search: a validated query in, real and cleaned job postings out.

    from jobscout.jobs import JobSearch, SearchQuery
    search = JobSearch.from_settings()              # one per agent run
    result = search.search(SearchQuery(text="data analyst in Dubai", country="ae"))
    result.jobs, result.cache_hit

Hidden inside: the JSearch HTTP call and its retries, the 24 h Redis cache, the
per-run and monthly quota guards, and turning raw JSearch data into JobPosting.
`filter_jobs` removes duplicates and postings that can't be a match.
"""

from ._client import JSearchError
from ._filter import FilterResult, FilterRules, filter_jobs
from ._models import DatePosted, EmploymentType, JobPosting, SearchQuery, SearchResult
from ._search import JobSearch, JSearchNotConfigured

__all__ = [
    "DatePosted",
    "EmploymentType",
    "FilterResult",
    "FilterRules",
    "JSearchError",
    "JSearchNotConfigured",
    "JobPosting",
    "JobSearch",
    "SearchQuery",
    "SearchResult",
    "filter_jobs",
]
