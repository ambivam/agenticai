"""Search tools for retrieving information from various sources"""

import logging
from typing import List, Dict, Any, Optional
import requests
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from config import Config

logger = logging.getLogger(__name__)

class SearchTools:
    """Collection of search tools for retrieving information"""
    
    def __init__(self):
        """Initialize search tools with necessary configurations"""
        self.google_api_key = Config.GOOGLE_API_KEY
        self.google_cse_id = Config.GOOGLE_CSE_ID
        # Wikipedia API endpoint
        self.wiki_api = 'https://en.wikipedia.org/w/api.php'
        
    def search_wikipedia(self, query: str, max_results: int = None) -> List[Dict[str, str]]:
        """Search Wikipedia for the given query"""
        try:
            if max_results is None:
                max_results = Config.WIKIPEDIA_RESULTS_COUNT
            
            results = []
            
            # Search Wikipedia
            search_params = {
                'action': 'query',
                'list': 'search',
                'format': 'json',
                'srsearch': query,
                'srlimit': max_results,
                'srprop': 'snippet'
            }
            
            search_response = requests.get(self.wiki_api, params=search_params)
            search_data = search_response.json()
            
            if 'query' in search_data and 'search' in search_data['query']:
                for result in search_data['query']['search']:
                    title = result['title']
                    
                    # Get page content
                    content_params = {
                        'action': 'query',
                        'format': 'json',
                        'titles': title,
                        'prop': 'extracts',
                        'exintro': True,
                        'explaintext': True
                    }
                    
                    content_response = requests.get(self.wiki_api, params=content_params)
                    content_data = content_response.json()
                    
                    try:
                        pages = content_data['query']['pages']
                        page = next(iter(pages.values()))
                        
                        if 'extract' in page:
                            # Get first 3 sentences of the extract
                            summary = '. '.join(page['extract'].split('.')[:3]) + '.'
                            
                            results.append({
                                'title': title,
                                'summary': summary,
                                'url': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
                                'source': 'Wikipedia'
                            })
                    except Exception as e:
                        logger.warning(f"Error processing Wikipedia page {title}: {str(e)}")
                        continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Wikipedia search: {str(e)}")
            return []
    
    def search_duckduckgo(self, query: str, max_results: int = None) -> List[Dict[str, str]]:
        """Search DuckDuckGo for the given query"""
        try:
            if max_results is None:
                max_results = Config.DUCKDUCKGO_RESULTS_COUNT
                
            results = []
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                for r in ddg_results:
                    results.append({
                        'title': r['title'],
                        'body': r['body'],
                        'url': r['link'],
                        'source': 'DuckDuckGo'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {str(e)}")
            return []
    
    def search_google(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Search Google using Custom Search API"""
        try:
            if not (self.google_api_key and self.google_cse_id):
                logger.warning("Google search credentials not configured")
                return []
                
            # Build Google Custom Search service
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            
            # Execute search
            result = service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=max_results
            ).execute()
            
            # Format results
            results = []
            if 'items' in result:
                for item in result['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'url': item.get('link', ''),
                        'source': 'Google'
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Google search: {str(e)}")
            return []
    
    def search_all(self, query: str) -> Dict[str, List[Dict[str, str]]]:
        """Search across all available search engines"""
        results = {
            'wikipedia': self.search_wikipedia(query),
            'duckduckgo': self.search_duckduckgo(query),
            'google': self.search_google(query) if self.google_api_key else []
        }
        return results
