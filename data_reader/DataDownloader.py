"""
Class to download and process SEC 8-K filings from EDGAR database
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import logging
from datetime import datetime
import time
from typing import List
import finnhub
import os



class UrlToText:

    def __init__(self):
        ...

class SECFillingsReader:

    def __init__(self, ticker):
        self.ticker = ticker
        self.supported_file_types = ['8-K']

    def filing_retrieval(self, ticker: str, start_date: str, end_date: str, filing_type: str):
        """
        Retrieve a list of SEC EDGAR filing links for a given ticker,
        within a specified date range, for a particular filing type.

        :param ticker: Ticker symbol (e.g., 'AAPL' for Apple).
        :param start_date: Start date "YYYY-MM-DD" for the search range.
        :param end_date: End date "YYYY-MM-DD" for the search range.
        :param filing_type: Filing type (e.g., '8-K', '10-K', '10-Q'). Currently only supporting 8-K. I left this
        broadly inorder to build more support for further development.

        :return: A list of dictionaries. An example out for each dictionary contains:
                 {
                    "accessNumber": "0001193125-20-039203",
                    "symbol": "AAPL",
                    "cik": "320193",
                    "form": "8-K",
                    "filedDate": "2020-02-18 00:00:00",
                    "acceptedDate": "2020-02-18 06:24:57",
                    "reportUrl": "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000119312520039203/d845033d8k.htm",
                    "filingUrl": "https://www.sec.gov/Archives/edgar/data/320193/000119312520039203/0001193125-20-039203-index.html"
                }
        """

        if filing_type not in self.supported_file_types:
            raise Exception("Only supporting 8-K File Types")

        output = []

        finnhub_client = finnhub.Client(api_key=os.environ['FINNHUB_API_KEY'])
        filings = finnhub_client.filings(symbol=ticker, _from=start_date, to=end_date)

        for filing_item in filings:
            if filing_item['form'] == filing_type:
                output.append(filing_type)

        return output



    def save_output_to_dir(self):


    def construct_url_links(self, filing_items: List):
        urls = []

        for item in filing_items:
            urls.append(item['reportUrl'])

        return urls


    def edgar_bulk_download(self):
        ...


    def build_txt_file_from_link(self):


    def time_window_generator(self, filing_item):
        """
        :param filing_item:
        :return:
        """
