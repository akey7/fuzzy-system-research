import os
import pandas as pd

data_store_filename = os.path.join(
    "/Volumes", "BlueData", "nasdaq_wiki_prices", "assests.h5"
)
wiki_prices_filename = os.path.join(
    "/Volumes", "BlueData", "nasdaq_wiki_prices", "wiki_prices.csv"
)

print(wiki_prices_filename)
print(data_store_filename)

df = pd.read_csv(wiki_prices_filename)
print(df.info(show_counts=True))
with pd.HDFStore(data_store_filename) as store:
    store.put("quandl/wiki/stocks", df)
