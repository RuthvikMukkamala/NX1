import os
import argparse
from backend.data_reader.DataDownloader import SECFillingsReader
from backend.model.EventClassification import LocalLLMEventClassification


class SECFilingProcessor:
    def __init__(self, ticker, start_date, end_date, event_categories):
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.event_categories = event_categories or [
            "Acquisition", "Customer Event", "Personnel Change", "Scheduling Event"
        ]
        self.sec_reader = SECFillingsReader(self.ticker)

    def fetch_filings(self):
        print(f"Fetching 8-K filings for {self.ticker} from {self.start_date} to {self.end_date}...")
        filings = self.sec_reader.filing_retrieval(self.ticker, self.start_date, self.end_date, "8-K")
        if not filings:
            print("No filings found for the given date range.")
            return []
        print("Saving filings to local directory...")
        self.sec_reader.save_output_to_dir(filings)
        return filings

    def process_filings(self, filings, method="cot", history=None):
        for filing in filings:
            file_path = os.path.join(self.ticker, "8K", f"{filing['accessNumber']}.txt")
            print(f"Processing: {file_path}")
            classifier = LocalLLMEventClassification(file_path, event_categories=self.event_categories)
            classifier.classify_events(method=method, history=history)


def parse_arguments():
    parser = argparse.ArgumentParser(description="SEC 8-K Filings Classifier")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--method", type=str, default="cot", choices=[
        "cot", "guided", "zero_shot", "few_shot", "self_consistency",
        "tree_of_thought", "self_refinement", "decomposition", "react"
    ], help="Select classification method")
    parser.add_argument("--history", type=int, default=None, help="Number of prompt history logs to display")
    parser.add_argument("--categories", type=str, nargs="+",
                        help="Custom event categories (space-separated list)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    processor = SECFilingProcessor(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        event_categories=args.categories
    )

    filings = processor.fetch_filings()
    if filings:
        processor.process_filings(filings, method=args.method, history=args.history)


if __name__ == "__main__":
    main()
