import requests
from bs4 import BeautifulSoup

import json

import pandas as pd
# For tidying up the code
from typing import Protocol, Callable, Any

# Defining custom variable types

type Dataframe = pd.DataFrame
type Data = list[dict[str, Any]]

import re

def parse_price(price_str):
    if not price_str:
        return None

    # Normalize whitespace (NBSP → space)
    cleaned = price_str.replace("\xa0", " ").lower()

    # Extract digits only
    digits = re.findall(r"\d+", cleaned)
    if not digits:
        return None

    return int("".join(digits))



# Currently not used
BASE_URL = "https://www.finn.no/"
SEARCH_PATH = "recommerce/forsale/search"

class Scraper(Protocol):
    def extract(self) -> Data:
        ...   
class Exporter(Protocol):
    def export(self, data) -> None:
        ...
        

class GetWebPageHtml:
    def __init__(self) -> None:
        self.scrape_date: str = str
        self.session = requests.Session()
        # Try to fool Finn, not sure it is working
        self.session.headers.update({ 
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "nb-NO,nb;q=0.9,en-US;q=0.8,en;q=0.7",
        })

    def post_query(self, url: str, query_params: dict[str: str|int]) -> Any:
        '''
        Docstring for posting a query
        
        Method for posting a query to a web page and returns the response object
        
        **Will break if the query raises an exception.**
        This can be the stopping condition, that there are no more pages available
        
        :param url: url to post query against
        :type url: str
        :param query_params: dict of parameters for the query
        :type query_params: dict[str: str]
        :return: Returns the response object
        :rtype: Any
        '''
        res = self.session.get(
            url=url,
            params=query_params,
            timeout=15
        )
        try:
            res.raise_for_status()
        except:
            return None
        
        return res
    
# Inherits a post_query method and a requests session
class FinnTorgetScraper(GetWebPageHtml):
    BASE_URL = "https://www.finn.no/"
    SEARCH_PATH_TORGET = "recommerce/forsale/search"
    URL: str = str # Will be set depending on what search method is used
    
    TORGET_SEARCH_PARAMS: dict[str, str|int] = {} 
    
    def __init__(self):
        # Initialise parsers
        self.html_parsers = {
            'JSON_LD_parser' : HtmlResponseParserAsJSON(),
            'DOM_parser' : HtmlResponseParserSelect()}
        
        # Select parser
        self.selected_parser : object = None # Must be set before 
        
        # Store parsed results
        self.JSON_LD_responses: list[dict[str, Any]] = []
        self.DOM_responses: list[dict[str, Any]] = []
        
        super().__init__() 
        
    
    
        
        
    # New scrape method:
    def scrape(self, search_word: str, parser: str = "JSON_LD_parser") -> list[dict]:
        url = f"{self.BASE_URL}{self.SEARCH_PATH_TORGET}"

        self.TORGET_SEARCH_PARAMS.update({
            "q": search_word,
            "page": 1,
            "price_from": None,
            "sort": "PRICE_ASC"
        })
        
        '''
        'q' : str,
        'page': 1,
        'sort': "PRICE_ASC", # RELEVANCE, PUBLISHED_ASC, PUBLISHED_DESC, PRICE_DESC, PRICE_ASC
        'price_from' : int, # Will this be interpreted as an empty field if it is passed as query parameter?
        'price_to': int,
        'trade_type': int # 1= Til salgs, 2=Gis bort, 3=Ønskes kjøpt
        '''

        # Select parser
        try:
            self.selected_parser = self.html_parsers[parser]
        except KeyError:
            raise KeyError("Parser must be JSON_LD_parser or DOM_parser")

        seen: set[str] = set()
        results: list[dict] = []

        # The idea is to incrementally increase the starting price until there are no more items left in the response
        # The crawl stops after a set of empty returns. An empty return could also just be an automatic stop
        price_from = 0
        MAX_EMPTY_WINDOWS = 3
        empty_windows = 0

        while empty_windows < MAX_EMPTY_WINDOWS:
            self.TORGET_SEARCH_PARAMS.update({
                "price_from": price_from,
                "page": 1,
            })

            window_items = []

            for item in self.page_crawler(url, self.TORGET_SEARCH_PARAMS):
                window_items.append(item)
                results.append(item)

            if not window_items:
                empty_windows += 1
                continue

            empty_windows = 0

            # Advance price window 
            if parser == 'JSON_LD_parser':
                last_price: str = window_items[-1]['offers'].get("price")
            
            elif parser == 'DOM_parser':
                last_price: str = window_items[-1].get("price")

            price_from = int(last_price) + 1
            print(f"Advancing price window from {last_price} to {price_from}. Found {len(window_items)} items this round. Empty windows: {empty_windows} ")

        return results

    def handle_search(self, url):
        # TODO: Implement search
        pass
    
    def page_crawler(
        self,
        url: str,
        query_params: dict[str, str | int],
    ):
        """
        Iterates over pages and yields individual items.
        Stops when:
        - no items are returned
        - max_pages is reached
        """

        page = 0

        while True:
            response = self.post_query(url=url, query_params=query_params)
            if not response:
                return
            items = self.selected_parser.parse(response)

            if not items:
                return

            for item in items:
                yield item

            page += 1
            query_params["page"] = page
            
                
        
            
    
    
# ---------------- Html Parsers ----------------------------------------# 
class HtmlResponseParserAsJSON:
    # Class that parses the Html response by targeting "script#seoStructuredData" in the html response
    def parse(self, response: object) -> list[dict[str, str | int]]:  # but document JSON-LD structure. TODO: Implement protocol
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        script = soup.select_one('script#seoStructuredData')
        if not script:
            print('WARNING: Found no data in the response. Returning empty list')
            return []        
        
        
        data_raw = script.string 
        if not data_raw:
            print("WARNING: No raw data from script")
        
        try:
            data = json.loads(data_raw)
        except:
            print("Failed to load script text to JSON")
            print(data_raw)
            return []
        
        if not data_raw or not data_raw.strip():
            print("WARNING: seoStructuredData is empty")
            return []

        try:
            data = json.loads(data_raw)
        except json.JSONDecodeError as e:
            print("WARNING: Invalid JSON-LD:", e)
            return []

        items_json = data["mainEntity"].get("itemListElement", [])
        if not items_json:
            print(f'Found no data in the response items json file. Items: {items_json}')
            return []
        
        # Add date scraped metadata
        items = [item['item'] for item in items_json] # Returns only the items in each article, not everything else
        
        return items
    
class HtmlResponseParserSelect:
    # Class tha parses the Html response by selecting certain elements.
    # Utilizes the bs4 method select() and get()
    
    def parse(self, response: object) -> list[dict[str: str | int]]:
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        items = []
        
        for article in soup.select("article.sf-search-ad"):
            
            a = article.select_one("a.sf-search-ad-link[href]")
            if not a: # No links in this listing. Must have a link
                continue
            
            items_dict = {}
            
            # Link to listing
            href = a.get("href")
            if href:
                items_dict['link'] = href
                
            # Title
            title = a.get_text(strip=True)
            if title:
                items_dict['title'] = title
                
            # Price
            price_span = article.select_one(
                "div.font-bold span"
            )
            price = price_span.get_text(strip=True) if price_span else None
            if price:
                items_dict['price_raw'] = price
                items_dict['price'] = parse_price(price)
                
            # Appending the results
            if items_dict:
                items.append(items_dict)
            
            
        return items
    
class JSONExporter:
    def __init__(self, filename: str):
        self.filename = filename

    def export(self, data: Data) -> None:
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=2)    
            

# Controller class for executing the actions in the right order
# TODO: Make sure all classes are speaking the same language
class DataPipeline:
    def __init__(self, scraper: Scraper, exporter: Exporter):
        self.scraper = scraper
        self.exporter = exporter

    def run_scrape_listings(self, search_word: str) -> None:
        # Scrape: Extract the listed items from finn.no
        data = self.scraper.scrape(search_word=search_word)
        
        # Export: Export the cleaned data 
        self.exporter.export(data)




# TODO: Ensure this is up to date when scraping logic is implemented
class Container: # Register all the different classes/components in this container so it can handle them all
    def __init__(self) -> None:
        self._providers: dict[str, tuple[Callable[[], Any], bool]] = {}
        self._singletons: dict[str, Any] = {} # If there is only supposed to be one instance of the class
        
    def register(self, name: str, provider: Callable[[], Any], singleton: bool = False) -> None: 
        self._providers[name] = (provider,singleton)
        
    def resolve(self, name: str) -> Any:
        if name in self._singletons:
            return self._providers[name]
        
        if name not in self._providers:
            raise ValueError(f"No provider registered for {name}")
        
        provider, singleton = self._providers[name]
        instance = provider()
        
        if singleton:
            self._singletons[name] = instance
        
        return instance



# TODO: Implement main function and how to run it

def main(search_word: str) -> None:
    container = Container()
    
    # TODO: Register all relevant classes
    output_filename = 'data/finn_listings_items.json'
    container.register("scraper", lambda: FinnTorgetScraper(), singleton=True)
    container.register("exporter", lambda: JSONExporter(output_filename))
    
    container.register("pipeline", lambda: DataPipeline(
        scraper=container.resolve('scraper'),
        exporter=container.resolve("exporter"),
    ))
    
    pipeline: DataPipeline = container.resolve("pipeline")
    pipeline.run_scrape_listings(search_word=search_word)
    print(f"Pipeline finished. Output written to {output_filename}")
    
    
if __name__ == __name__:
    main(search_word = "gitar")
