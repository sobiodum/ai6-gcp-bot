# File: data_management/data_fetcher.py

import pandas as pd
from typing import Dict, List, Optional
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timedelta
import pytz
import logging


class OandaDataFetcher:
    """Handles data fetching from OANDA API with robust error handling and rate limiting."""

    def __init__(
            self,
            account_id: str,
            access_token: str
    ):
        """Initialize the OANDA API client."""
        self.account_id = account_id
        self.api = API(access_token=access_token)
        self.logger = logging.getLogger(__name__)

        # Standard timeframe definitions
        self.timeframes = {
            "1min": "M1",
            "5min": "M5",
            "15min": "M15",
            "1h": "H1"
        }
        # Maximum candles per request based on timeframe
        self.max_chunks = {
            "1min": timedelta(hours=4),    # ~240 candles
            "5min": timedelta(days=1),     # ~288 candles
            "15min": timedelta(days=2),    # ~192 candles
            "1h": timedelta(days=7)        # ~168 candles
        }

    def fetch_candles(
        self,
        instrument: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        now = datetime.now(pytz.UTC)

        if start_time is None:
            start_time = now - timedelta(days=30)
        elif not start_time.tzinfo:
            start_time = pytz.UTC.localize(start_time)

        if end_time is None:
            end_time = now
        elif not end_time.tzinfo:
            end_time = pytz.UTC.localize(end_time)

        if end_time > now:
            end_time = now

        fetch_time = start_time
        candles_data = []
        chunk_size = self.max_chunks[timeframe]

        while fetch_time < end_time:
            try:
                batch_end = min(fetch_time + chunk_size, end_time)

                request = instruments.InstrumentsCandles(
                    instrument=instrument,
                    params={
                        "granularity": self.timeframes[timeframe],
                        "price": "M",
                        "from": fetch_time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                        "to": batch_end.strftime("%Y-%m-%dT%H:%M:%S.000000Z")
                    }
                )

                response = self.api.request(request)
                if not response.get("candles"):
                    # If no candles in response, still advance the time
                    print('no response')
                    fetch_time = batch_end
                    continue
                print(response)

                for candle in response["candles"]:
                    if candle["complete"]:
                        candles_data.append({
                            "timestamp": pd.Timestamp(candle["time"]),
                            "open": float(candle["mid"]["o"]),
                            "high": float(candle["mid"]["h"]),
                            "low": float(candle["mid"]["l"]),
                            "close": float(candle["mid"]["c"]),
                            "volume": int(candle["volume"])
                        })

                fetch_time = batch_end

            except Exception as e:
                self.logger.error(f"Error fetching data: {str(e)}")
                raise

        df = pd.DataFrame(candles_data)
        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        return df
