import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import random
from collections import defaultdict
import pickle 

from utils import XRetrieval, sample_balanced_edits, save_chunks, filter_by_time_and_format_time_columns, filter_for_added_len_of_columns

from global_config import TARGET_START_DATE_X, TARGET_END_DATE_X, SEED, N_SAMPLES_PER_MONTH_PER_CAT, MAX_SAMPLE_INFO_LENGTH_CHARS

DATA_FOLDER = "X_input_data"
FILENAME_NOTES = "notes-00000.tsv"
FILENAME_NOTES_HISTORY = "noteStatusHistory-00000.tsv"
FILENNAME_RATINGS = "ratings-0000{}.tsv"
N_RATING_SHEETS = 7

random.seed(SEED)
np.random.seed(SEED)


def retrieve_n_sampled_notes(notes_df, n_samples, X_retriever):

    # randomly change ordering of rows 
    notes_df = notes_df.sample(frac=1).reset_index(drop=True)

    relevant_indices = []
    total_retrieved = 0

    # initialize columns
    notes_df["XLink"] = ""
    notes_df["able_to_retrieve"] = False
    notes_df["XText"] = ""
    notes_df["XhasMedia"] = False
    notes_df["uses_quote"] = False
    notes_df["quoteText"] = ""
    notes_df["quoteHasMedia"] = False
    notes_df["language"] = ""
    notes_df["is_relevant"] = False

    n_able_to_retrieve = 0
    n_lan_eng = 0
    n_no_media = 0
    n_no_quote_media = 0
    n_perm_length = 0


    for index, note in tqdm(notes_df.iterrows()):

        X_link = f"https://twitter.com/anyuser/status/{int(note['tweetId'])}"

        able_to_retrieve, X_text, language, has_media, uses_quote, quote_text, quote_has_media, _, _ = X_retriever.get_X(note["tweetId"])
        total_retrieved += 1

        total_len = len(X_text) + len(quote_text) + len(note["NoteText"])

        if able_to_retrieve:
            n_able_to_retrieve += 1
            if language == "en":
                n_lan_eng += 1
            if not has_media:
                n_no_media += 1
            if not quote_has_media:
                n_no_quote_media += 1
            if total_len <= MAX_SAMPLE_INFO_LENGTH_CHARS:
                n_perm_length += 1

        if able_to_retrieve and language == "en" and not has_media and not quote_has_media and total_len <= MAX_SAMPLE_INFO_LENGTH_CHARS:
            is_relevant = True
            relevant_indices.append(index)
        else:
            is_relevant = False

        notes_df.loc[index, "XLink"] = X_link
        notes_df.loc[index, "able_to_retrieve"] = able_to_retrieve
        notes_df.loc[index, "XText"] = X_text
        notes_df.loc[index, "XhasMedia"] = has_media
        notes_df.loc[index, "uses_quote"] = uses_quote
        notes_df.loc[index, "quoteText"] = quote_text
        notes_df.loc[index, "quoteHasMedia"] = quote_has_media
        notes_df.loc[index, "language"] = language
        notes_df.loc[index, "is_relevant"] = is_relevant


        if len(relevant_indices) >= n_samples:
            break

    print(f"Retrieved {total_retrieved} notes to get {len(relevant_indices)} samples")
    print(f"Able to retrieve: {n_able_to_retrieve/total_retrieved}")
    print(f"for retrieved notes: English: {round(n_lan_eng/n_able_to_retrieve, 1)}; No media: {round(n_no_media/n_able_to_retrieve, 1)}; No quote media: {round(n_no_quote_media/n_able_to_retrieve, 1)}; Permissible length: {round(n_perm_length/n_able_to_retrieve, 1)}")
    return notes_df, relevant_indices


def select_and_rename_columns(notes_df):
    columns_to_keep = ["noteId", "summary", "tweetId", "helpful",
                       "Date and Time", "Date and Time String", "MonthYear", "WeekOfYear"]
    
    notes_df = notes_df[columns_to_keep].copy()

    notes_df = notes_df.rename(columns={"summary": "NoteText"})
    notes_df["ID"] = notes_df["noteId"]

    return notes_df


def retrieve_raw_data_Xs(output_folder):

    X_retriever = XRetrieval()

    notes = pd.read_csv(os.path.join(DATA_FOLDER, FILENAME_NOTES), sep='\t', low_memory=False)

    notes_history = pd.read_csv(os.path.join(DATA_FOLDER, FILENAME_NOTES_HISTORY), sep='\t', low_memory=False)

    notes["Date and Time"] = notes["createdAtMillis"].apply(lambda x: datetime.datetime.fromtimestamp(x/1000))
    notes = filter_by_time_and_format_time_columns(notes, TARGET_START_DATE_X, TARGET_END_DATE_X) 

    notes_history_helpful = notes_history[notes_history["currentStatus"]=="CURRENTLY_RATED_HELPFUL"].copy()
    notes_helpful = notes[notes["noteId"].isin(notes_history_helpful["noteId"])].copy()
    notes_helpful["helpful"] = True

    notes_history_nothelpful = notes_history[notes_history["currentStatus"]=="CURRENTLY_RATED_NOT_HELPFUL"].copy()
    notes_nothelpful = notes[notes["noteId"].isin(notes_history_nothelpful["noteId"])].copy()
    notes_nothelpful["helpful"] = False

    print(f" Total number of recent notes with helpful status: {len(notes_helpful)}")
    print(f" Total number of recent notes with not helpful status: {len(notes_nothelpful)}")

    notes_helpful = select_and_rename_columns(notes_helpful)
    notes_nothelpful = select_and_rename_columns(notes_nothelpful)

    notes_helpful_df, relevant_indices_helpful = retrieve_n_sampled_notes(notes_helpful, N_SAMPLES_PER_MONTH_PER_CAT, X_retriever)
    notes_nothelpful_df, relevant_indices_nothelpful = retrieve_n_sampled_notes(notes_nothelpful, N_SAMPLES_PER_MONTH_PER_CAT, X_retriever)

    notes_helpful_df_relevant = notes_helpful_df.loc[relevant_indices_helpful].copy()
    notes_nothelpful_df_relevant = notes_nothelpful_df.loc[relevant_indices_nothelpful].copy()

    all_notes_relevant = pd.concat([notes_helpful_df_relevant, notes_nothelpful_df_relevant], ignore_index=True)

    all_notes_relevant.to_pickle(os.path.join(output_folder, "all_notes_relevant.pkl"))
    notes_helpful_df.to_pickle(os.path.join(output_folder, "all_helpful.pkl"))
    notes_nothelpful_df.to_pickle(os.path.join(output_folder, "all_nothelpful.pkl"))

    all_notes_relevant.to_excel(os.path.join(output_folder, "all_notes_relevant.xlsx"))
    notes_helpful_df.to_excel(os.path.join(output_folder, "all_helpful.xlsx"))
    notes_nothelpful_df.to_excel(os.path.join(output_folder, "all_nothelpful.xlsx"))

    X_retriever.save_tweet_dict()


def load_votes_by_noteid(is_old=False):
    if not is_old:
        votes_loaded = pickle.load(open(os.path.join(DATA_FOLDER, "votes_by_noteid.pkl"), "rb"))
    else:
        votes_loaded = pickle.load(open(os.path.join(DATA_FOLDER, "votes_by_noteid_old.pkl"), "rb"))
        
    return votes_loaded

if __name__ == "__main__":

    TARGET_START_DATE_X.strftime('%d-%m-%Y')

    assert TARGET_START_DATE_X.day == 1
    assert TARGET_END_DATE_X.day == 1
    assert TARGET_END_DATE_X.month == (TARGET_START_DATE_X.month + 1) % 12

    time_id = TARGET_START_DATE_X.strftime('%m-%Y')

    output_folder = os.path.join(DATA_FOLDER, time_id)
    os.makedirs(output_folder, exist_ok=True)

    retrieve_raw_data_Xs(output_folder)

    all_notes = pd.read_pickle(os.path.join(output_folder, "all_notes_relevant.pkl"))

    print("total number of helpful notes: ", len(all_notes[all_notes["helpful"]==True]))
    print("total number of not helpful notes: ", len(all_notes[all_notes["helpful"]==False]))

    # # Format and filter the dataset
    all_notes = filter_by_time_and_format_time_columns(all_notes, TARGET_START_DATE_X, TARGET_END_DATE_X) 
    all_notes = filter_for_added_len_of_columns(all_notes, ["XText", "quoteText", "NoteText"], MAX_SAMPLE_INFO_LENGTH_CHARS)

    # Sample balanced datasets for both helpful and non-helpful edits
    all_notes = sample_balanced_edits(all_notes, time_column="MonthYear")

    if not len(all_notes) == 2 * N_SAMPLES_PER_MONTH_PER_CAT:
        print("Error: Did not get the expected number of samples. Only have ", len(all_notes), " samples.")
        print("number of helpful notes: ", len(all_notes[all_notes["helpful"]==True]))
        print("number of not helpful notes: ", len(all_notes[all_notes["helpful"]==False]))
    

    save_chunks(all_notes, dataset_id="X")

