from extract_relevant_from_jsonl import extract_relevant_from_jsonl
from get_close_data import get_close_data
from get_close_data import concat_close_data


# you would use it like this:

# get SEC data from https://api.sec-api.io/bulk/form-4/year/yyyy-mm.jsonl.gz?token=YOUR_API_KEY 
# i.e. https://api.sec-api.io/bulk/form-4/2018/2018-01.jsonl.gz?token=YOUR_API_KEY  <- insert Key

# for each year and month file do:
# yyyy = ["2009","2010","2011","2012","2013","2014","2015","2016","2017","2018","2019","2020","2021","2022","2023","2024"]
# mm = ["01","02","03","04","05","06","07","08","09","10","11","12"]

# for y in yyyy:
#    for m in mm:
#        filename = f"{y}-{m}"
#        extract_relevant_from_jsonl(month=filename)
#        get_close_data(month=filename) #  or get_close_data(month=filename, horizon_days= 365)

# concat_close_data(data_dir = "data/2026_01")







