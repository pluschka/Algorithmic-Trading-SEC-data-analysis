from extract_relevant_from_jsonl import extract_relevant_from_jsonl
from get_close_data import get_close_data
from get_close_data import concat_close_data


# you would use it like this:

# get data from https://api.sec-api.io/bulk/form-4/year/yyyy-mm.jsonl.gz?token=YOUR_API_KEY 
# i.e. https://api.sec-api.io/bulk/form-4/2018/2018-01.jsonl.gz?token=YOUR_API_KEY  <- insert Key

# for each year and month file do:
# extract_relevant_from_jsonl(filename='2018-01')
# get_close_data(filename='2018-01')

# adjust in get_close_data.py the list of data according to your file names and then do:
# all_df_of_close_data, expected_final_row_number = concat_close_data()