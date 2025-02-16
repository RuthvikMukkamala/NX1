import requests
from bs4 import BeautifulSoup
from typing import List
import finnhub
import os


class SECFillingsReader:

    def __init__(self, ticker):
        self.ticker = ticker
        self.supported_file_types = ['8-K']
        self.email = 'ruthvikmukkamala@nyu.edu'
        self.headers = {
            'User-Agent': f'Mozilla/5.0 (compatible; SECFilingsReader/1.0; {self.email})',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }

    def filing_retrieval(self, ticker: str, start_date: str, end_date: str, filing_type: str):
        if filing_type not in self.supported_file_types:
            raise Exception("Only supporting 8-K File Types")

        output = []

        finnhub_client = finnhub.Client(api_key=os.environ['FINNHUB_API_KEY'])
        filings = finnhub_client.filings(symbol=ticker, _from=start_date, to=end_date)

        for filing_item in filings:
            if filing_item['form'] == filing_type:
                output.append(filing_item)

        return output

    def save_output_to_dir(self, filing_items: List):
        output_dir = os.path.join(self.ticker, "8K")
        os.makedirs(output_dir, exist_ok=True)

        for item in filing_items:
            file_name = f"{item['accessNumber']}.txt"
            file_path = os.path.join(output_dir, file_name)
            self.build_txt_file_from_link(item['reportUrl'], file_path)

    def construct_url_links(self, filing_items: List):
        return [item['reportUrl'] for item in filing_items]

    def build_txt_file_from_link(self, link: str, output_path: str) -> None:
        try:
            response = requests.get(link, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(separator="\n")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"Saved {link} to {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve {link}. Error: {e}")
