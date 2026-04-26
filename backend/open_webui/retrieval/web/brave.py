import logging
import time
from typing import Optional

import requests
from open_webui.retrieval.web.main import SearchResult, get_filtered_results

log = logging.getLogger(__name__)


def search_brave(
    api_key: str,
    query: str,
    count: int,
    filter_list: Optional[list[str]] = None,
    country: Optional[str] = None,
    search_lang: Optional[str] = None,
    spellcheck: Optional[bool] = None,
    maximum_number_of_tokens: int = 8192,
    maximum_number_of_tokens_per_url: int = 4096,
    maximum_number_of_snippets_per_url: int = 50,
    context_threshold_mode: str = 'balanced',
    freshness: Optional[str] = None,
    goggles: Optional[str] = None,
) -> list[SearchResult]:
    """Search using Brave's LLM Context API and return the results as a list of SearchResult objects.

    Args:
        api_key (str): A Brave Search API key
        query (str): The query to search for
        count (int): Maximum number of URLs to include in the context
        filter_list (list[str], optional): Domain allow/block list
        country (str, optional): 2-letter country code to localise results
        search_lang (str, optional): Language code for search results
        spellcheck (bool, optional): Whether to enable query spellcheck
        maximum_number_of_tokens (int): Approximate max tokens in context (default 8192, max 32768)
        maximum_number_of_tokens_per_url (int): Max tokens per URL (default 4096, max 8192)
        maximum_number_of_snippets_per_url (int): Max snippets per URL (default 50, max 100)
        context_threshold_mode (str): Threshold mode — 'balanced' or 'aggressive' (default 'balanced')
        freshness (str, optional): Age filter — pd/pw/pm/py or YYYY-MM-DDtoYYYY-MM-DD
        goggles (str, optional): Goggle URL or definition to rerank results
    """

    url = 'https://api.search.brave.com/res/v1/llm/context'
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': api_key,
    }
    params = {
        'q': query,
        'maximum_number_of_urls': count,
        'maximum_number_of_tokens': maximum_number_of_tokens,
        'maximum_number_of_tokens_per_url': maximum_number_of_tokens_per_url,
        'maximum_number_of_snippets_per_url': maximum_number_of_snippets_per_url,
        'context_threshold_mode': context_threshold_mode,
    }

    # Only include optional params when explicitly set
    if country:
        params['country'] = country
    if search_lang:
        params['search_lang'] = search_lang
    if spellcheck is not None:
        params['spellcheck'] = spellcheck
    if freshness:
        params['freshness'] = freshness
    if goggles:
        params['goggles'] = goggles

    response = requests.get(url, headers=headers, params=params)

    # Handle 429 rate limiting - Brave free tier allows 1 request/second
    # If rate limited, wait 1 second and retry once before failing
    if response.status_code == 429:
        log.info('Brave Search API rate limited (429), retrying after 1 second...')
        time.sleep(1)
        response = requests.get(url, headers=headers, params=params)

    response.raise_for_status()

    json_response = response.json()
    results = json_response.get('grounding', {}).get('generic', [])
    if filter_list:
        results = get_filtered_results(results, filter_list)

    return [
        SearchResult(
            link=result['url'],
            title=result.get('title'),
            snippet='\n\n'.join(result.get('snippets', [])),
        )
        for result in results[:count]
    ]
    