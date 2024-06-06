import pandas as pd
import ast
import datetime
import random
import numpy as np
import os

from global_config import TARGET_START_DATE_WIKI, TARGET_END_DATE_WIKI, MAX_SAMPLE_INFO_LENGTH_CHARS, SEED
import datetime

from utils import sample_balanced_edits, save_chunks, filter_by_time_and_format_time_columns, filter_for_added_len_of_columns

random.seed(SEED)
np.random.seed(SEED)


DATA_FOLDER = "wiki_input_data"
FILENAME_KEPT = "wiki_all_kept_Feb_March_24.csv"
FILENAME_REVERTED = "wiki_all_reverted_Feb_March_24.csv"

def load_and_prepare_data(file_path, helpful_flag):
    """Load, format, and filter the dataset."""
    df = pd.read_csv(file_path)
    df["helpful"] = helpful_flag
    df["ID"] = df["revid"]
    df["Date and Time"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ"))
    df['paragraph'] = df['paragraph'].apply(lambda x: ' '.join(ast.literal_eval(x)))

    # filter out all pages that have "talk" in the title
    print(f"Number of rows before filtering for talk: {len(df)}")
    df = df[~df["title"].str.contains("talk", case=False)]
    print(f"Number of rows after filtering for talk: {len(df)}")
    
    return df

# Load and prepare datasets
wiki_kept = load_and_prepare_data(os.path.join(DATA_FOLDER, FILENAME_KEPT), helpful_flag=True)
wiki_reverted = load_and_prepare_data(os.path.join(DATA_FOLDER, FILENAME_REVERTED), helpful_flag=False)
wiki_all = pd.concat([wiki_kept, wiki_reverted]).reset_index(drop=True)

# Format and filter the dataset
wiki_all = filter_by_time_and_format_time_columns(wiki_all, TARGET_START_DATE_WIKI, TARGET_END_DATE_WIKI)
wiki_all = filter_for_added_len_of_columns(wiki_all, ["paragraph", "deleted", "added"], MAX_SAMPLE_INFO_LENGTH_CHARS)
    
# Sample balanced datasets for both helpful and non-helpful edits
wiki_all = sample_balanced_edits(wiki_all, time_column="WeekOfYear")

save_chunks(wiki_all, dataset_id="wiki")
