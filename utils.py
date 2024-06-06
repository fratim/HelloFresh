
import os
import requests
import pickle
import os
import time
import traceback 
import pandas as pd

from global_config import CHUNK_SIZE, TEST_CHUNK_SIZE

import time
import requests
import re
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import string
from bs4 import BeautifulSoup
import traceback
import math
import numpy as np

from global_config import N_ARTICLES_LIMIT

import sys

PRINTABLE_CHARS = set(string.printable)

def clean_string(input_string):
    # Replace multiple spaces with a single space
    input_string = re.sub(' +', ' ', input_string)
    # Replace multiple newlines or tabs with a single newline
    input_string = re.sub('\n+', '\n', input_string)
    input_string = re.sub('\t+', '\n', input_string)
    # remove all sorts of LLM stuff
    input_string = input_string.replace("<s>", "").replace("</s>", "").replace("<<SYS>>", "")
    input_string = input_string.replace("<</SYS>>", "").replace("[INST]", "").replace("[/INST]", "").replace("<<", "")
    input_string = input_string.replace(">>", "").replace("...", "").replace("  ", " ").replace('<unk>', '')
    input_string = input_string.strip()
    #remove illegal characters
    input_string = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', input_string)
    input_string = ILLEGAL_CHARACTERS_RE.sub(r'', input_string)
    
    return input_string


def clean_dataframe(df):

    for col in df.columns:
        for i in range(len(df)):

            if isinstance(df.loc[i, col], str):
                df.loc[i, col] = ''.join(filter(lambda x: x in PRINTABLE_CHARS, df.loc[i, col]))

    return df


def is_valid_article(elem):
    return isinstance(elem, str) and len(elem)>2000


def fill_df_elems(df, df_positions, df_elems):
    for elem_i, df_position in enumerate(df_positions):
        row_i, col_name = df_position
        df.loc[row_i, col_name] = df_elems[elem_i]
    return df


def get_page_links(query, dataset):

    if dataset == "X":
        blocked_urls = ["twitter.com", "X.com"]
    elif dataset == "wiki":
        blocked_urls = ["wikipedia.org"]
    else:
        raise ValueError("Invalid dataset")

    def _get_links(query):
        url = "https://google-api31.p.rapidapi.com/websearch"

        payload = {
            "text": query,
            "safesearch": "off",
            "timelimit": "10",
            "region": "wt-wt",
            "max_results": int(N_ARTICLES_LIMIT*5),
        }
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": "none",
            "X-RapidAPI-Host": "google-api31.p.rapidapi.com"
        }

        response_original = requests.post(url, json=payload, headers=headers)
        result_json = response_original.json()["result"]

        page_links = [result["href"] for result in result_json if not any([url in result["href"] for url in blocked_urls])]

        return page_links, result_json

    query = str(query).replace('"', '')
    iteration = 0
    n_iterations = 5
    error_message = "--no error message--"
    sleep_time = 2
    last_result_json = None

    while iteration < n_iterations:
        try:
            page_links, result_json = _get_links(query)
            last_result_json = result_json
            if len(page_links) > 0:
                break
    
        except Exception as e:  # Capture the exception as 'e'
            error_message = traceback.format_exc()
            iteration += 1
            time.sleep(sleep_time)
            sleep_time *= 2

    if iteration == n_iterations:
        page_links = []

    if len(page_links) < N_ARTICLES_LIMIT:
        print(f"Warning: only {len(page_links)} links found for query {query}")
        print(f"EXCEPTION: {error_message}")
        print(f"RESULT JSON: {last_result_json}")

    return page_links


def get_page_contents(links):
    
    def _get_page_content(link):
        iteration = 0
        n_iterations = 5
        sleep_time = 2
        while iteration < n_iterations:
            try:
                page_response = requests.get(link, timeout=5)
                page_content = BeautifulSoup(page_response.text, 'html.parser')
                content_text = page_content.get_text()
                return content_text
            except Exception as e:  # Capture the exception as 'e'
                iteration += 1
                time.sleep(sleep_time)
                sleep_time *= 2

        if iteration == n_iterations:
            return ""

    page_contents = []
    for link in links:
        
        # skip link if it ends with a file extension
        if link.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx')):
            continue

        content_text = _get_page_content(link)
        content_text = clean_string(content_text)

        if is_valid_article(content_text):
            page_contents.append(content_text)
        
        if len(page_contents) >= N_ARTICLES_LIMIT:
            break

    if len(page_contents) < N_ARTICLES_LIMIT:
        print(f"Warning: only {len(page_contents)} articles found for links {links}")

    return page_contents


def get_results_folder(dataset, month, model_type, prompt_type, chunk_idx, summaries_config):
    output_folder = f"{dataset}_results/{month}/{model_type}/{prompt_type}/{summaries_config}/chunk{chunk_idx}"
    os.makedirs(output_folder, exist_ok=True)

    return output_folder


def filter_by_time_and_format_time_columns(df, start_time, end_time):
    """Filter and add necessary columns to the dataframe."""

    # Filter rows based on date range and create a copy to avoid warnings
    df = df[(df['Date and Time'] >= start_time) & (df['Date and Time'] < end_time)].copy()

    df["Date and Time String"] = df["Date and Time"].dt.strftime('%d-%m-%Y %H:%M:%S')
    df["MonthYear"] = df["Date and Time"].dt.strftime('%B-%Y')
    df["WeekOfYear"] = df["Date and Time"].dt.isocalendar().week

    return df

def filter_for_added_len_of_columns(df, column_names, max_total_length):

    print(f"Number of data samples before filtering for total char length: {len(df)}")
    for column in column_names:
        df.loc[:, f'{column}_len'] = df[column].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df['total_len'] = df[[f'{column}_len' for column in column_names]].sum(axis=1)
    df = df[df['total_len'] < max_total_length]

    print(f"Number of data samples after filtering for total char length: {len(df)}")

    return df

    
def sample_balanced_edits(df, time_column):
    """Sample an equal number of helpful and non-helpful edits per week."""

    print(f"Number of data samples before balancing: {len(df)}")

    balanced_dfs = []
    for timesspan in df[time_column].unique():
        timespan_df = df[df[time_column] == timesspan]

        if len(timespan_df["helpful"].unique()) != 2:
            continue

        min_class_count = timespan_df["helpful"].value_counts().min()
        
        balanced_timespan_df = timespan_df.groupby("helpful").sample(n=min_class_count, replace=False).reset_index(drop=True)
        balanced_dfs.append(balanced_timespan_df)
    
    final_df = pd.concat(balanced_dfs).sample(frac=1).reset_index(drop=True)
    
    print(f"Number of data samples after balancing: {len(final_df)}")
    
    return final_df


def save_chunks(data_all, dataset_id):

    assert data_all["MonthYear"].nunique() == 1
    monthyear_str = data_all["MonthYear"].iloc[0]

    output_dir = f"{dataset_id}_data/{monthyear_str}/blank/"
    os.makedirs(output_dir, exist_ok=True)

    helpful_edits = data_all[data_all["helpful"] == True]
    non_helpful_edits = data_all[data_all["helpful"] == False]
    assert len(helpful_edits) == len(non_helpful_edits)

    # Alternate helpful and non-helpful edits
    final_dataset_list = []
    for i in range(len(helpful_edits)):
        final_dataset_list.append(helpful_edits.iloc[i])
        final_dataset_list.append(non_helpful_edits.iloc[i])

    # Convert the list of tuples back into a DataFrame
    final_dataset = pd.DataFrame(final_dataset_list, columns=helpful_edits.columns)

    # Split and save dataset into chunks

    num_chunks = math.ceil(len(final_dataset) / CHUNK_SIZE)
    print(f"Number of chunks: {num_chunks}")
    for i in range(num_chunks):
        chunk = final_dataset[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE].copy().reset_index(drop=True)
        print(len(chunk))
        chunk.to_pickle(os.path.join(output_dir, f"chunk{i}.pkl"))

    # Save a test chunk
    test_chunk = final_dataset[:TEST_CHUNK_SIZE].copy().reset_index(drop=True)
    test_chunk.to_pickle(os.path.join(output_dir, "chunk99.pkl"))


class XRetrieval:

    def __init__(self):
        self.tweet_folder = "retrieved_X_data"
        self.tweet_dict_fname = "X_dict_most_recent"
        
        self.tweet_dict = self.load_tweet_dict()

        # delete all keys for which "Rate limit exceeded." is in the value

        self.tweet_dict = {k: v for k, v in self.tweet_dict.items() if "Rate limit exceeded." not in str(v)}

        self.save_freq = 50
        self.print_freq = 500
        
        self.n_loaded = 0
        self.n_api_calls = 0
        self.n_timeouts = 0
        self.n_abletoretrieve = 0

        self.url = "https://twttrapi.p.rapidapi.com/get-tweet"

        self.headers = {
            "X-RapidAPI-Key": "none",
            "X-RapidAPI-Host": "twttrapi.p.rapidapi.com"
        }


    def get_tweet_dict_fpath(self, timestamp=None):
        if timestamp is None:
            return os.path.join(self.tweet_folder, self.tweet_dict_fname + ".pkl")
        else:
            return os.path.join(self.tweet_folder, self.tweet_dict_fname + "_" + timestamp + ".pkl")


    def load_tweet_dict(self):        

        if not os.path.exists(self.get_tweet_dict_fpath()):
            print(f"Could not find tweet dict at {self.get_tweet_dict_fpath()}")
            print("Creating new tweet dict")
            return {}
        else:
            with open(self.get_tweet_dict_fpath(), "rb") as f:
                tweet_dict = pickle.load(f)
            return tweet_dict
        

    def write_dict(self, tweet_dict, fpath):
        fpath_temp = fpath + "_temp"
        with open(fpath_temp, "wb") as f:
            pickle.dump(tweet_dict, f)
        os.rename(fpath_temp, fpath)

    def save_tweet_dict(self):
        
        self.write_dict(self.tweet_dict, self.get_tweet_dict_fpath())

        timestr = str(time.time()).split(".")[0]
        self.write_dict(self.tweet_dict, self.get_tweet_dict_fpath(timestr))


    def save_response_json(self, tweet_id, response_json):
        self.tweet_dict[tweet_id] = response_json

        if self.n_api_calls % self.save_freq == 0:
            self.save_tweet_dict()


    def get_response_json(self, tweet_id):

        if (self.n_loaded + self.n_api_calls) % self.print_freq == 0 or (self.n_api_calls > 0 and self.n_api_calls % self.save_freq == 0):
            print(f"n_total: {self.n_loaded + self.n_api_calls}, "\
                f"n_abletoretrieve: {self.n_abletoretrieve}, "\
                f"n_loaded: {self.n_loaded}, "\
                f"n_api_calls: {self.n_api_calls}, "\
                f"n_timeouts: {self.n_timeouts}")

        if tweet_id in self.tweet_dict.keys():
            
            response_json = self.tweet_dict[tweet_id]
            self.n_loaded += 1

            return response_json
            
        else:
            self.n_api_calls += 1

            response_original = requests.get(self.url, headers=self.headers, params={"tweet_id": str(tweet_id)})

            max_it = 10
            sleep_time = 0.5
            current_it = 0

            while 'Rate limit exceeded.' in str(response_original.content):
                print(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                response_original = requests.get(self.url, headers=self.headers, params={"tweet_id": str(tweet_id)})
                current_it += 1
                sleep_time *= 2
                if current_it == max_it:
                    break

            if response_original.status_code == 504:
                response_json = None
                self.n_timeouts += 1
            else:
                response_json = response_original.json()
                self.save_response_json(tweet_id, response_json)
                
            return response_json


    def get_X(self, tweet_id, debug=False):

        response_json = self.get_response_json(tweet_id)

        if "Rate limit exceeded." in str(response_json):
            raise ValueError("Rate limit exceeded.")
            
        if debug:
            print(response_json)

        try:  
            response_result = response_json['data']['tweet_result']['result']

            if 'tweet' in response_result:
                response_result = response_result['tweet']

            language = response_result["legacy"]["lang"]

            if "note_tweet" in response_result:
                tweet_text = response_result['note_tweet']['note_tweet_results']['result']['text']
            else:
                tweet_text = response_result['legacy']['full_text']

            if  "extended_entities" in response_result["legacy"] or "tweet_card" in response_result:
                has_media = True
            else:
                has_media = False

            if "quoted_status_result" in response_result and len(response_result['quoted_status_result']) > 0:
                uses_quote = True

                if "tweet" in response_result['quoted_status_result']['result']:
                    response_result['quoted_status_result']['result'] = response_result['quoted_status_result']['result']['tweet']
                    
                quote_text = response_result['quoted_status_result']['result']['legacy']['full_text']

                if "extended_entities" in response_result["quoted_status_result"]['result']["legacy"] or "tweet_card" in response_result["quoted_status_result"]['result']:
                    quote_has_media = True
                else:
                    quote_has_media = False
            else:
                uses_quote = False
                quote_text = ""
                quote_has_media = False

            able_to_retrieve = True
            response_json = response_json
            error_message = ""

        except Exception as e:  # Capture the exception as 'e'
            error_message = traceback.format_exc()

            if len(str(response_json)) > 1000:
                print(f"Could not retrieve tweet with id {int(tweet_id)}")
                print(e)
                print(response_json)                

            tweet_text = ""
            has_media = False
            able_to_retrieve = False
            response_json = response_json
            error_message = error_message
            uses_quote = False
            quote_text = ""
            quote_has_media = False
            language = ""

        if able_to_retrieve:
            self.n_abletoretrieve += 1

        return able_to_retrieve, tweet_text, language, has_media, uses_quote, quote_text, quote_has_media, response_json, error_message

if __name__ == "__main__":
    X_retriever = XRetrieval()

    if len(sys.argv) == 1:
        X_id = "1681656310230818817"
    else:
        X_id = int(sys.argv[1])

    print(f"Getting X for tweet id {X_id}")
    able_to_retrieve, tweet_text, language, has_media, uses_quote, quote_text, quote_has_media, response_json, error_message = X_retriever.get_X(X_id, debug=True)

    print(f"able_to_retrieve: {able_to_retrieve}")
    print(f"tweet_text: {tweet_text}")
