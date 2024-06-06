import wandb
import pandas as pd
from tqdm import tqdm
import argparse
import os
from api_helpers import get_completions
from utils import get_results_folder
import random
import numpy as np

from utils import clean_string, clean_dataframe, is_valid_article, fill_df_elems, get_page_contents, get_page_links

from input_strings import SEARCH_QUERY_PROMPT, SUMMARIZATION_PROMPT, CLASSIFICATION_WO_SUMMARIES_PROMPT, CLASSIFICATION_WITH_SUMMARIES_PROMPT, RANDOM_SUMMARIES, REFUSED_RESPONSE_PREFIXES

from global_config import N_ARTICLES_LIMIT, DEBUG, DEBUG_SAMPLES, SEED

random.seed(SEED)
np.random.seed(SEED)

class Eval_Handler():

    def __init__(self, model_type, dataset, month, prompt_type):
        
        self.model_type = model_type
        self.dataset = dataset
        self.month = month
        self.prompt_type = prompt_type

        self.max_summary_length_char = 1000 
        self.max_summary_length_tok = 250 

        self.max_searchquery_length_tok = 30

        self.max_article_length_char = 4000
    

    def get_X_note_info(self, X_txt, uses_quote, quote_txt, note_txt, datetime):
        
        X_info = f"""SOCIAL MEDIA POST ({datetime.date()}): {X_txt}"""

        if uses_quote:
            quote_info = f""" quoting {quote_txt}"""

        note_info = f"""NOTE: {note_txt}"""

        if uses_quote:
            return f"{X_info}{quote_info} \n \n{note_info} \n \n"
        else:
            return f"{X_info} \n \n{note_info} \n \n"


    def get_wiki_edit_info(self, title, text_paragraph, text_deleted, text_added, datetime):

        splitted_title = title.split("#")
        page_title = splitted_title[0].replace("_", " ")
        subsection = splitted_title[1].replace("_", " ") if len(splitted_title) > 1 else ""

        text_deleted = text_deleted if isinstance(text_deleted, str) else ""
        text_added = text_added if isinstance(text_added, str) else ""

        section_string = f", section '{subsection}'" if subsection else ""

        paragraph_info = f"""ARTICLE: {page_title}{section_string} \n \nDate of Edit: {datetime.date()} \n \nPARAGRAPH: {text_paragraph}"""

        deleted_info = f"""PROPOSED DELETION: {text_deleted if len(text_deleted) > 0 else "None"}"""
        
        added_info = f"""PROPOSED ADDITION: {text_added if len(text_added) > 0 else "None"}"""

        return f"{paragraph_info} \n \n{deleted_info} \n \n{added_info} \n \n"


    def get_searchqueriers(self, datasample_infos):

        prompts = []
        
        for datasample_info in datasample_infos:
            prompt = datasample_info + SEARCH_QUERY_PROMPT[self.dataset]

            prompts.append(prompt)

        search_prompts = get_completions(prompts, model_type=self.model_type, max_new_tokens=self.max_searchquery_length_tok)
        search_prompts = [elem.replace('"', '').replace("\n", "") for elem in search_prompts]

        if DEBUG: 
            print(*[f"prompt: {prompt} \n completion: {search_prompt}" for prompt, search_prompt in zip(prompts[:DEBUG_SAMPLES], search_prompts[:DEBUG_SAMPLES])], sep='\n \n ')

        return search_prompts
    

    def get_summaries(self, inputs):

        prompts = []
                
        for article_text, datasample_info in inputs:
            
            article_text = fit_text_to_n_chars(article_text, n_chars=self.max_article_length_char)
            prompt = f""" PROVIDED INFORMATION: {article_text} \n """ + datasample_info + SUMMARIZATION_PROMPT[self.dataset]

            prompts.append(prompt)

        article_summaries = get_completions(prompts, model_type=self.model_type, max_new_tokens=self.max_summary_length_tok)
        article_summaries = [fit_text_to_n_chars(elem, n_chars=self.max_summary_length_char) for elem in article_summaries]


        if DEBUG:
            print(*[f" prompt: {prompt} \n completion: {summary}"  for prompt, summary in zip(prompts[:DEBUG_SAMPLES], article_summaries[:DEBUG_SAMPLES])], sep='\n \n ')

        return article_summaries
    

    def get_classifications(self, inputs):
        
        prompts = []

        for summaries, datasample_info in inputs:
            
            use_summaries = True if summaries is not None else False

            if use_summaries:
                summaries_info = "".join(f"SUMMARY {i+1}: {summaries[i]} \n " for i in range(len(summaries)))
                instruction_info = CLASSIFICATION_WITH_SUMMARIES_PROMPT[self.dataset][self.prompt_type]    
                prompt = summaries_info + datasample_info + instruction_info
            else: 
                instruction_info = CLASSIFICATION_WO_SUMMARIES_PROMPT[self.dataset][self.prompt_type]
                prompt = datasample_info + instruction_info

            prompts.append(prompt)

        completions = get_completions(prompts, model_type=self.model_type, max_new_tokens=15)
        classifications = [get_classifiction_from_completion(elem) for elem in completions]

        if DEBUG:
            print(*[f" prompt: {prompt} \n completion: {completion} \n classification: {classification}"  for prompt, completion, classification in zip(prompts[:DEBUG_SAMPLES], completions[:DEBUG_SAMPLES], classifications[:3])], sep='\n \n ')
                  
        return classifications


def fit_text_to_n_chars(input_text, n_chars):

    assert isinstance(input_text, str), "input_text must be a string"

    if len(input_text) > n_chars:
        text = input_text[:n_chars]
        text = clean_string(text)
        text += " \n "
    else:
        text = input_text

    return text


def get_classifiction_from_completion(completion):
    
    if any([completion.startswith(prefix) for prefix in REFUSED_RESPONSE_PREFIXES]):
        return_value = "response blocked"
    elif "response blocked" in completion:
        return_value = "response blocked"
    elif "yes" in completion[:5] or "Yes" in completion[:5]:
        return_value = "yes"
    elif "no" in completion[:5] or "No" in completion[:5]:
        return_value = "no"
    else:
        return_value =  "none"

    return return_value


def get_input_fname(step, dataset, month, model_type, chunk_idx):
    if step == "save_searchqueriers":
        input_folder = f"{dataset}_data/{month}/blank"
        input_fname = os.path.join(input_folder, f"chunk{chunk_idx}.pkl")
    elif step == "save_articles":
        input_folder = f"{dataset}_data/{month}/searchqueriers/{model_type}"
        input_fname = os.path.join(input_folder, f"chunk{chunk_idx}.pkl")
    elif step == "save_summaries":
        input_folder = f"{dataset}_data/{month}/articles/{model_type}"
        input_fname = os.path.join(input_folder, f"chunk{chunk_idx}.pkl")
    elif step == "save_class_wo_summ":
        input_folder = f"{dataset}_data/{month}/blank"
        input_fname = os.path.join(input_folder, f"chunk{chunk_idx}.pkl")
    elif step == "save_class_with_summ":
        input_folder = f"{dataset}_data/{month}/summaries/{model_type}"
        input_fname = os.path.join(input_folder, f"chunk{chunk_idx}.pkl")
    elif step == "save_class_with_rand_summ":
        input_folder = f"{dataset}_data/{month}/blank"
        input_fname = os.path.join(input_folder, f"chunk{chunk_idx}.pkl")
    else:
        raise ValueError(f"step {step} not recognized") 
    return input_fname


def get_output_fname(step, dataset, month, model_type, chunk_idx):
    if step == "save_searchqueriers":
        output_dir = f"{dataset}_data/{month}/searchqueriers/{model_type}"
        output_fname = os.path.join(output_dir, f"chunk{chunk_idx}.pkl")
    elif step == "save_articles":
        output_dir = f"{dataset}_data/{month}/articles/{model_type}"
        output_fname = os.path.join(output_dir, f"chunk{chunk_idx}.pkl")
    elif step == "save_summaries":
        output_dir = f"{dataset}_data/{month}/summaries/{model_type}"
        output_fname = os.path.join(output_dir, f"chunk{chunk_idx}.pkl")
    else:
        raise ValueError(f"step {step} not recognized") 
    
    os.makedirs(output_dir, exist_ok=True)

    return output_fname


def write_results_to_csvfile(results_output_fname, IDs, TimestampStrs, predicted_labels, true_labels):

    df_id = pd.DataFrame(IDs, columns=['ID'])
    df_timestamp = pd.DataFrame(TimestampStrs, columns=['TimstampStr'])
    df_predicted_labels = pd.DataFrame(predicted_labels, columns=['predicted_labels'])
    df_true_labels = pd.DataFrame(true_labels, columns=['true_labels'])

    df_results = pd.concat([df_id, df_timestamp, df_predicted_labels, df_true_labels], axis=1)
    df_results.to_csv(results_output_fname, index=False)

    print("wrote results to results folder")


def get_datasample_info(dataset, row_elem):

    if dataset == "X":
        X_note_info = eval_handler.get_X_note_info(row_elem["XText"], row_elem["uses_quote"], row_elem["quoteText"], row_elem["NoteText"], row_elem["Date and Time"])
        return X_note_info
    elif dataset == "wiki":
        wiki_edit_info = eval_handler.get_wiki_edit_info(row_elem["title"], row_elem["paragraph"], row_elem["deleted"], row_elem["added"], row_elem["Date and Time"])
        return wiki_edit_info
    else:
        raise ValueError(f"dataset {dataset} not recognized")


def save_searchqueriers(eval_handler, chunk_idx):

    input_fname = get_input_fname("save_searchqueriers", eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)
    output_fname = get_output_fname("save_searchqueriers", eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)

    if os.path.isfile(output_fname):
        print(f"chunk {chunk_idx}: searchqueriers already saved")
        return

    print(f"chunk {chunk_idx}: saving searchqueriers")

    data_frame = pd.read_pickle(input_fname)

    df_elemts_to_compute_positions = []
    df_elemts_to_compute_inputs = []

    for row_i, row_elem in tqdm(data_frame.iterrows()):
        
        data_sample_info = get_datasample_info(eval_handler.dataset, row_elem)

        df_elemts_to_compute_positions.append((row_i, "searchquery"))
        df_elemts_to_compute_inputs.append(data_sample_info)

    df_elemts_to_compute_outputs = eval_handler.get_searchqueriers(df_elemts_to_compute_inputs)
    
    data_frame = fill_df_elems(data_frame, df_elemts_to_compute_positions, df_elemts_to_compute_outputs)

    data_frame = clean_dataframe(data_frame)
    data_frame.to_pickle(output_fname)

    if wandb.run:
        wandb.log({"saved_searchqueriers": True}, commit=True)


def save_articles(eval_handler, chunk_idx):
    
    input_fname = get_input_fname("save_articles", eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)
    output_fname = get_output_fname("save_articles", eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)

    if os.path.isfile(output_fname):
        print(f"chunk {chunk_idx}: articles already saved")
        return

    data_frame = pd.read_pickle(input_fname)

    notes_with_insufficient_articles = 0

    for row_i, row_elem in tqdm(data_frame.iterrows()):

        page_links = get_page_links(row_elem["searchquery"], eval_handler.dataset)
        page_contents = get_page_contents(page_links)

        if len(page_contents) < N_ARTICLES_LIMIT:
            notes_with_insufficient_articles += 1

        for result_i, (link, content) in enumerate(zip(page_links, page_contents)):
            data_frame.loc[row_i, f"link_{result_i}"] = link
            data_frame.loc[row_i, f"article_{result_i}"] = clean_string(content)
    
    data_frame = clean_dataframe(data_frame)
    data_frame.to_pickle(output_fname)

    print(f"notes_with_insufficient_articles: {notes_with_insufficient_articles}")

    if wandb.run:
        wandb.log({"saved_articles": True,
                   "n_with_ins_articles": notes_with_insufficient_articles}, commit=True)

def save_summaries(eval_handler, chunk_idx):
    
    input_fname = get_input_fname("save_summaries", eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)
    output_fname = get_output_fname("save_summaries", eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)

    if os.path.isfile(output_fname):
        print(f"chunk {chunk_idx}: summaries already saved")
        return

    print(f"chunk {chunk_idx}: saving summaries")

    data_frame = pd.read_pickle(input_fname)

    df_elemts_to_compute_positions = []
    df_elemts_to_compute_inputs = []

    for row_i, row_elem in tqdm(data_frame.iterrows()):
        
        data_sample_info = get_datasample_info(eval_handler.dataset, row_elem)

        for i, article_i in enumerate([row_elem[f"article_{i}"] for i in range(N_ARTICLES_LIMIT)]):
            if is_valid_article(article_i):
                df_elemts_to_compute_positions.append((row_i, f"article_summary_{i}"))
                df_elemts_to_compute_inputs.append((article_i, data_sample_info))

    df_elemts_to_compute_outputs = eval_handler.get_summaries(df_elemts_to_compute_inputs)

    data_frame = fill_df_elems(data_frame, df_elemts_to_compute_positions, df_elemts_to_compute_outputs)
    
    data_frame = clean_dataframe(data_frame)
    data_frame.to_pickle(output_fname)

    if wandb.run:
        wandb.log({"saved_summaries": True}, commit=True)


def save_classifications(eval_handler, chunk_idx, summaries_config):

    if summaries_config == "none":
        use_summaries = False
        rand_summ = False
        step_name = "save_class_wo_summ"
    elif summaries_config == "regular":
        use_summaries = True
        rand_summ = False
        step_name = "save_class_with_summ"
    elif summaries_config == "random":
        use_summaries = True
        rand_summ = True
        step_name = "save_class_with_rand_summ"
    else:
        raise ValueError(f"summaries_config {summaries_config} not recognized")

    input_fname = get_input_fname(step_name, eval_handler.dataset, eval_handler.month, eval_handler.model_type, chunk_idx)

    results_output_fname = os.path.join(get_results_folder(dataset=eval_handler.dataset,
                                                           month=eval_handler.month,
                                                           model_type=eval_handler.model_type, 
                                                           prompt_type=eval_handler.prompt_type, 
                                                           chunk_idx=chunk_idx, 
                                                           summaries_config=summaries_config), "results.csv")
    
    if os.path.isfile(results_output_fname):
        print(f"chunk {chunk_idx} - summaries {summaries_config}: classifications already saved")
        return

    data_frame = pd.read_pickle(input_fname)

    df_elemts_to_compute_positions = []
    df_elemts_to_compute_inputs = []

    for row_i, row_elem in tqdm(data_frame.iterrows()):

        data_sample_info = get_datasample_info(eval_handler.dataset, row_elem)

        df_elemts_to_compute_positions.append((row_i, "classification_with_summaries" if use_summaries else "classification_wo_summaries"))

        if use_summaries:
            if rand_summ:
                summaries = RANDOM_SUMMARIES
            else:
                summaries = [row_elem[f"article_summary_{i}"] for i in range(N_ARTICLES_LIMIT)]
        else:
            summaries = None

        df_elemts_to_compute_inputs.append((summaries, data_sample_info))

    df_elemts_to_compute_outputs = eval_handler.get_classifications(df_elemts_to_compute_inputs)

    data_frame = fill_df_elems(data_frame, df_elemts_to_compute_positions, df_elemts_to_compute_outputs)
    
    write_results_to_csvfile(
        results_output_fname=results_output_fname, 
        IDs=data_frame["ID"].tolist(), 
        TimestampStrs=data_frame["Date and Time String"].tolist(), 
        predicted_labels=df_elemts_to_compute_outputs, 
        true_labels=data_frame["helpful"].tolist())

    if wandb.run:
        wandb.log({f"{step_name}": True}, commit=True)


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_idx", type=int, help="Chunk index", required=True)
    parser.add_argument("--function", choices=['all', \
                                               'searchqueriers', \
                                               'articles', \
                                                'summaries', \
                                                "class_wo_summ", \
                                                "class_with_summ", \
                                                "class_with_rand_summ" ], help="Function to execute", required=True)
    parser.add_argument("--model_type", choices=['GPT4', 'GPT3.5', 'LLAMA70B', 'GEMINI', "MISTRAL87"], help="which model to evaluate", required=True)
    parser.add_argument("--dataset", choices=['X', 'wiki'], help="which dataset to evaluate on", required=True)
    parser.add_argument("--prompt_type", choices=['MANUAL', 'GPT4', 'GPT3.5', 'MISTRAL87', 'LLAMA70B'], help="which prompt to use", default="MANUAL")
    parser.add_argument("--month", help="for which month to load the data", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()
    print("reached this point")

    # Do not need eval handler (LLM models) if we we just scrape articles
    eval_handler = Eval_Handler(model_type=args.model_type,
                                dataset=args.dataset,
                                month=args.month,
                                prompt_type=args.prompt_type)

    # decide which step of the pipeline to run
    if args.function in ['all', 'searchqueriers']:
        save_searchqueriers(eval_handler, chunk_idx=args.chunk_idx)

    if args.function in['all', 'articles']:
        save_articles(eval_handler, chunk_idx=args.chunk_idx)
    
    if args.function in ['all', 'summaries']:
        save_summaries(eval_handler, chunk_idx=args.chunk_idx)

    if args.function in ['all', 'class_wo_summ']:
        save_classifications(eval_handler, chunk_idx=args.chunk_idx, summaries_config="none")

    if args.function in ['all', 'class_with_summ']:
        save_classifications(eval_handler, chunk_idx=args.chunk_idx, summaries_config="regular")

    if args.function in ['class_with_rand_summ']:
        save_classifications(eval_handler, chunk_idx=args.chunk_idx, summaries_config="random")
