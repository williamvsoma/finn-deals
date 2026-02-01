import time
import requests
from bs4 import BeautifulSoup

from datetime import datetime

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



class Extractor(Protocol):
    def extract(self) -> Data:
        ...
        
class Transformer(Protocol):
    def transform(self, data) -> Data:
        ...
        
class Parser(Protocol):
    def parse(self, response) -> list[dict[str, str | int]]:
        ...
        
class Exporter(Protocol):
    def export(self, data) -> None:
        ...
        

class GetWebPageHtml:
    def __init__(self) -> None:
        self.scrape_date: str = str
        session = requests.Session()
        # Try to fool Finn, not sure it is working
        session.headers.update({ 
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
        
        res.raise_for_status()
        
        return res
    
# Inherits a post_query method and a requests session
class FinnScraper(GetWebPageHtml):
    BASE_URL = "https://www.finn.no/"
    SEARCH_PATH_TORGET = "recommerce/forsale/search"
    URL: str = str # Will be set depending on what search method is used
    
    TORGET_SEARCH_PARAMS: dict[str, str|int] = {
        'q' : str,
        'page': 1,
        'sort': "PRICE_ASC", # RELEVANCE, PUBLISHED_ASC, PUBLISHED_DESC, PRICE_DESC, PRICE_ASC
        'price_from' : int, # Will this be interpreted as an empty field if it is passed as query parameter?
        'price_to': int,
        'trade_type': int # 1= Til salgs, 2=Gis bort, 3=Ønskes kjøpt
    } 
    
    def __init__(self):
        # Initialise parsers
        self.JSON_LD_parser = HtmlResponseParserAsJSON()
        self.DOM_parser = HtmlResponseParserSelect()
        
        # Store parsed results
        self.JSON_LD_responses: list[dict[str, Any]] = []
        self.DOM_responses: list[dict[str, Any]] = []
    

    def search_torget(self, search_word: str) -> Dataframe:
        # Method for handling a search to torget
        
        url = f"{self.BASE_URL}{self.SEARCH_PATH_TORGET}"
        
        self.TORGET_SEARCH_PARAMS['q'] = search_word
        
        # TODO: Should i construct the query parameters first, or try to construct them somewhat dynamically to ensure all the listing are retrieved?
        
        # What about parsing all the listing from the lowest price and ascending. When the crawler stops, retrieve the price of the last item and use this price (minus 1) as the starting price for the next crawl. Then check for duplicated saved entries at the end
        
        query_params = self.TORGET_SEARCH_PARAMS
        
        # TODO: Implement search
        
        # Initial crawl
        responses = 
        
    def handle_search(self, url):
        # TODO: Implement search
        pass
    
    def page_crawler(self, url: str, query_params: dict[str, str|int]) -> list[object] :
        
        '''
        Docstring for page_crawler
        
        :param self: Description
        :param url: Description
        :type url: str
        :param query_params: Description
        :type query_params: dict[str, str | int]
        :return: list of response objects
        :rtype: list[object]
        '''
        
        # method for crawling the pages based on the other search parameters
        
        responses: list[dict[str, str | int]] = []
        condition = True
        while condition:
            
            try: # Should stop when the max page limit is reached
                response = self.post_query(url=url, query_params=query_params)
            except:# TODO: Currently stops at page 50. My browser also stops at 50 and tells me to narrow down my search
                print(f"Got bad response, stopping crawl at page {query_params['page']}")
                condition = False
                
            
            responses.extend(self.JSON_LD_parser(response))
            query_params['page'] += 1
            
        parsed_responses = 
        return responses
                
                
            
    
    

class HtmlResponseParserAsJSON:
    # Class that parses the Html response by targeting "script#seoStructuredData" in the html response
    def parse(self, response: object) -> list[dict[str, str | int]]:  # but document JSON-LD structure. TODO: Implement protocol
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        script = soup.select_one('script#seoStructuredData')
        if not script:
            print('WARNING: Found no data in the response. Returning empty list')
            return []        
        
        
        data_raw = script.string or script.get_text() # Some sites can be text
        
        data = json.loads(data_raw)
        
        items_json = data["mainEntity"].get("itemListElement", [])
        
        # Add date scraped metadata

        items = [item['item'] for item in items_json] # Returns only the items in each article, not everything else
        
        if not items_json:
            raise ValueError(f'Found no data in the response items json file. Items: {items_json}')
        
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
class DataPipelineListings:
    def __init__(self, extractor: Extractor, parser: Parser, transformer: Transformer, exporter: Exporter):
        self.extractor = extractor
        self.transformer = transformer
        self.exporter = exporter

    def run(self) -> None:
        # Extract: Extract the listed items from finn.no
        
        # Transform: Filter out the data we want. Price, pictures, name, links to the listed item
        
        # Export: Export the cleaned data 
        pass
        




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
        
        if name not in self._singletons:
            raise ValueError(f"No provider registered for {name}")
        
        provider, singleton = self._providers[name]
        instance = provider()
        
        if singleton:
            self._singletons[name] = instance
        
        return instance

session = requests.Session()

# Try to fool Finn, not sure it is working
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept-Language": "nb-NO,nb;q=0.9,en-US;q=0.8,en;q=0.7",
})


query = "gitar"
page = 1
items_raw = []
scraped_at = datetime.today().date().isoformat()

while True:
    res = session.get(
        "https://www.finn.no/recommerce/forsale/search",
        params={"q": "gitar", "page": page},
        timeout=15,
    )
    try:
        res.raise_for_status()
        
    except: # TODO: Currently stops at page 50. My browser also stops at 50 and tells me to narrow down my search
        print('Got bad response: Saving results an exiting')

        with open("data/finn_items.jsonl", "a", encoding="utf-8") as f: # Storing only price, title and link
            for el in items_raw:
                item = el['item']
                item = {
                    "title": item["name"],
                    "price": int(item["offers"]["price"]),
                    "link": item["url"],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                
        with open("data/finn_items_raw.jsonl", "a", encoding="utf-8") as f: # Storing the raw items
            for el in items_raw:
                el['_meta'] = {  # Add metadata to the item
                    "query": query,
                    "page": page,
                    "scraped_at": scraped_at,
                    "source": "finn.no",
                }
                f.write(json.dumps(el, ensure_ascii=False) + "\n")
                
        print('Results saved -- Done')

        
        
    soup = BeautifulSoup(res.text, "html.parser")

    script = soup.select_one('script#seoStructuredData')
    if not script:
        break

    data = json.loads(script.string)
    items_json = data["mainEntity"].get("itemListElement", [])

    if not items_json:
        print("Final page reached.")
        break

    items = [el for el in items_json]
    

    print(f"Found {len(items)} listings on page {page}")
    items_raw.extend(items)
    page += 1
    
    time.sleep(1.5)
    
    if page == 49:
        time.sleep(20)
    



print('Searching done. Writing results')

with open("data/finn_items.jsonl", "a", encoding="utf-8") as f: # Storing only price, title and link
    for el in items_raw:
        item = el['item']
        item = {
            "title": item["name"],
            "price": int(item["offers"]["price"]),
            "link": item["url"],
        }
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        
with open("data/finn_items_raw.jsonl", "a", encoding="utf-8") as f: # Storing the raw items
    for el in items_raw:
        el['_meta'] = {  # Add metadata to the item
            "query": query,
            "page": page,
            "scraped_at": scraped_at,
            "source": "finn.no",
        }
        f.write(json.dumps(el, ensure_ascii=False) + "\n")
        
print('Results saved -- Done')


# TODO: Implement main function and how to run it

def main() -> None:
    container = Container()
    
    # TODO: Register all relevant classes
    
    container.register("extractor", FinnScraper(), singleton=True)
    container.register("exporter", lambda: JSONExporter('finn_listings_items.json'))
    
