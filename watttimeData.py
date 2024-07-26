# linux or mac
import os 
os.environ['WATTTIME_USER'] = 'nomanbashir'
os.environ['WATTTIME_PASSWORD'] = 'vikner-bozda4-fesrEb'

import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from watttime import WattTimeMyAccess
from watttime import WattTimeForecast
from watttime import WattTimeHistorical

wt_forecast = WattTimeForecast()

wt_historical = WattTimeHistorical()

wt_myaccess = WattTimeMyAccess()

carbon_data = [
    ("ap-southeast-2", "NEM_NSW"),
    ("ca-central-1", "HQ"),
    ("eu-central-1", "DE"),
    ("eu-north-1", "SE"),
    ("eu-west-2", "UK"),
    ("eu-west-3", "FR"),
    ("us-east-1", "PJM_DC"),
    ("us-west-2", "PACW"),
    ("us-west-1", "CAISO_NORTH"),
]

def downloadData(tuple):
    csv_name, region = tuple
    print("getting data for", csv_name)
    # load csv from carbon-data
    df = pd.read_csv(f"carbon-data/{csv_name}.csv", parse_dates=["datetime"])

    # filter data to only consider 2022 data
    df = df[df['datetime'] >= pd.Timestamp("2022-01-01", tz='UTC')]

    # just take the first 10 rows for testing
    # df = df.head(10)
    
    # get the list of timestamps in the underlying data
    timestamps = df['datetime']
    
    hist_marginal = []
    hist_marginal_forecast = []
    for time in tqdm(timestamps):
        start_time = time
        end_time = time + pd.Timedelta('59 minutes')

        model = "2022-10-01"
        # if start_time < pd.Timestamp("2020-03-01", tz='UTC'):
        #     model = "2022-10-01"

        # get the data from WattTime
        try:
            hist_hour = wt_historical.get_historical_pandas(
                start = start_time,
                end = end_time,
                region = region,
                signal_type = 'co2_moer',
                # model = model,
            )
            hist_forecasts_hour = wt_forecast.get_historical_forecast_pandas(
                start = start_time,
                end = end_time,
                region = region,
                signal_type = 'co2_moer',
                model = model,
            )
        except:
            # if it fails, wait a minute, try again.
            time.sleep(60)
            # try again
            try:
                hist_hour = wt_historical.get_historical_pandas(
                    start = start_time,
                    end = end_time,
                    region = region,
                    signal_type = 'co2_moer',
                    # model = model,
                )
                hist_forecasts_hour = wt_forecast.get_historical_forecast_pandas(
                    start = start_time,
                    end = end_time,
                    region = region,
                    signal_type = 'co2_moer',
                    model = model,
                )
            except:
                print("failed to get data for", time, " and ", region)
                hist_marginal.append(hist_marginal[-1])
                hist_marginal_forecast.append(hist_marginal_forecast[-1])
                continue
        avg_marginal = hist_hour['value'].mean()
        avg_forecast = hist_forecasts_hour['value'].mean()
        hist_marginal.append(avg_marginal)
        hist_marginal_forecast.append(avg_forecast)
    df['marginal_carbon_avg'] = hist_marginal
    df['marginal_forecast_avg'] = hist_marginal_forecast
    df.to_csv(f"marginal-data/{csv_name}.csv", index=False)
    print(f"Processed {csv_name}")

if __name__ == "__main__":
    # use multiprocessing to download data in parallel
    with Pool(5) as p:
        p.map(downloadData, carbon_data)
