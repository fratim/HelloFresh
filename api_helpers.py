import time
import replicate
import traceback
import time
import tqdm
from openai import OpenAI
from tqdm import tqdm
from utils import clean_string

import google.generativeai as genai
GEMINI_API_KEY="none"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

OAI_ACCESS_TOKEN = "none"
oai_client = OpenAI(api_key=OAI_ACCESS_TOKEN)


MODEL_NAMES = {"GPT3.5": "gpt-3.5-turbo-1106",
               "GPT4": "gpt-4-0125-preview",
               "LLAMA7B": "meta-llama/Llama-2-7b-chat-hf",
               "LLAMA70B": "meta-llama/Llama-2-70b-chat-hf",
               "GEMINI": "GEMINI",
               "MISTRAL87": "mistralai/Mixtral-8x7B-Instruct-v0.1"}


REPLICTE_SYSTEM_PROMPTS = {"meta/llama-2-70b-chat": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                  "mistralai/mixtral-8x7b-instruct-v0.1": "You are a very helpful, respectful and honest assistant."}

REPLICATE_PROMPT_TEMPLATE = {"meta/llama-2-70b-chat": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
                             "mistralai/mixtral-8x7b-instruct-v0.1": "<s>[INST] {prompt} [/INST] "}

def get_completions(input_prompts, model_type, max_new_tokens):
    
    assert isinstance(input_prompts, list)
    assert all(isinstance(prompt, str) for prompt in input_prompts)

    completions = []
    for input_prompt in tqdm(input_prompts):
        completions.append(get_completion(input_prompt, model_type, max_new_tokens))
    
    return completions


def get_completion(input_prompt, model_type, max_new_tokens):

    assert isinstance(input_prompt, str)

    iteration = 0
    n_iterations_max = 10
    wait_time = 5
    while iteration < n_iterations_max:
        completion = _get_completion_single(input_prompt, model_type, max_new_tokens)
        if completion.startswith("Exception:"):
            iteration += 1
            print(completion)
            print(f"Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            wait_time *= 2
        else:
            return clean_string(completion)
    
    if iteration == n_iterations_max:
        raise Exception("Max number of iterations reached. No completion was returned.")


def _get_completion_single(input_prompt, model_type, max_new_tokens):

    if model_type == "GPT3.5" or model_type == "GPT4":
        completion = get_completion_single_gpt(input_prompt, model_name=MODEL_NAMES[model_type], max_tokens=max_new_tokens)
        
    elif model_type == "GEMINI":
        completion = get_completion_single_gemini(input_prompt, max_tokens=max_new_tokens)
        
    elif model_type == "LLAMA70B" or model_type == "MISTRAL87":
        completion = get_completion_single_replicate(input_prompt, model_name=MODEL_NAMES[model_type], max_tokens=max_new_tokens)
        
    else:
        raise ValueError("Model not recognized.")
    
    return completion


def get_completion_single_gemini(prompt, max_tokens):
    # NOTE: cannot set max_tokens otherwise the API breaks
    try:
        response = gemini_model.generate_content(prompt,generation_config=genai.types.GenerationConfig(candidate_count=1,  temperature=0)) # , max_output_tokens=max_tokens*3))
        if response.prompt_feedback.block_reason.value == 0 and response.candidates[0].finish_reason.value in [0, 1, 2]:
            response_txt_full = response.candidates[0].content.parts[0].text
            response_txt_max_len = response_txt_full[:max_tokens*4]
            return response_txt_max_len
        else:
            print(f"Block reason: {str(response.prompt_feedback.block_reason)}")
            print(f"Finish reason: {str(response.candidates[0].finish_reason)}") if len(response.candidates) > 0 else None
            return "response blocked"
        
    except Exception as e:
        return f"Exception: An error occurred: {e}"


def get_completion_single_gpt(prompt, model_name, max_tokens):

    try:

        messages=[
                {"role": "user", "content": f"{prompt}"}
            ]
        
        response = oai_client.chat.completions.create(
            model=model_name,  # Specify the model to use
            messages=messages,  
            temperature=0,
            max_tokens=max_tokens, 
            n=1,  # Number of completions to generate
            logprobs=None, 
        )

        response = response.choices[0].message.content

        return response
        
    except Exception as e:
        return f"Exception: An error occurred: {e}"

    
def get_completion_single_replicate(prompt, model_name, max_tokens):



    if model_name == "meta-llama/Llama-2-70b-chat-hf":
        model_name_replicate = "meta/llama-2-70b-chat"
    elif model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        model_name_replicate = "mistralai/mixtral-8x7b-instruct-v0.1"
    else:
        raise ValueError("Model not recognized.")

    try:
        output = ""
        
        input = {
                "top_p": 0.9,
                "prompt": prompt,
                "temperature": 0.01,
                "system_prompt": REPLICTE_SYSTEM_PROMPTS[model_name_replicate],
                "prompt_template": REPLICATE_PROMPT_TEMPLATE[model_name_replicate],
                "max_new_tokens": max_tokens
                }
        
        output = replicate.run(model_name_replicate, input=input)
        output = "".join(output)

        return output

    except Exception as e:
        return f"Exception: An error occurred: {e}"


if __name__ == "__main__":
    print("Testing get_completion function...")

    print("Testing GPT-3.5...")
    print(get_completion("write me a poem", "GPT3.5", 50))
    
    print("Testing GPT-4...")
    print(get_completion("write me a poem", "GPT4", 50))
    
    print("Testing LLAMA-70B...")
    print(get_completion("write me a poem", "LLAMA70B", 50))
    
    print("Testing MISTRAL87...")
    print(get_completion("write me a poem", "MISTRAL87", 50))

    print("Testing GEMINI...")
    print(get_completion("write me a poem", "GEMINI", None))
    