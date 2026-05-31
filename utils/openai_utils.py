from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_exponential, after_log
from tqdm import tqdm
import openai
import logging
import sys

logger = logging.getLogger('customized_openai')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def extract_answer_from_thinking(text):
    try:
        text = text.split('</think>')[-1].strip()
        return text
    except:
        return text
    
def process_response(res, stream=False, completion=False, n=1):
    if stream:
        full_output_list = []
        for i in range(n):
            full_output_list.append("")

        last_finish_reason_list = [None for _ in range(n)]
        for chunk in tqdm(res, mininterval=2):
            for i in range(len(chunk.choices)):
                chunk_output = chunk.choices[i].delta.content

                if chunk_output is not None:
                    full_output_list[i] += chunk_output

                last_finish_reason_list[i] = chunk.choices[i].finish_reason

        print(f"Streaming Finish Reason: {last_finish_reason_list}")

        for i in range(len(last_finish_reason_list)):
            if last_finish_reason_list[i] != "stop":
                full_output_list[i] = "<INCOMPLETE_CONTENT>" + full_output_list[i]

        output = full_output_list
    else:
        output = []
        if not completion:
            for i in range(len(res.choices)):
                output.append(res.choices[i].message.content)
        else:
            for i in range(len(res.choices)):
                output.append(res.choices[i].text)

    return output

def process_response_seperate(res, stream=False, n=1, reasoning_chain_key='reasoning_content'):
    if stream:
        full_reasoning_list = []
        full_answer_list = []
        for i in range(n):
            full_reasoning_list.append("")
            full_answer_list.append("")

        last_finish_reason_list = [None for _ in range(n)]
        for chunk in tqdm(res, mininterval=2):
            for i in range(len(chunk.choices)):
                if hasattr(chunk.choices[i].delta, reasoning_chain_key) and getattr(chunk.choices[i].delta, reasoning_chain_key) is not None:
                    full_reasoning_list[i] += getattr(chunk.choices[i].delta, reasoning_chain_key)

                if hasattr(chunk.choices[i].delta, 'content') and chunk.choices[i].delta.content is not None:
                    full_answer_list[i] += chunk.choices[i].delta.content

                last_finish_reason_list[i] = chunk.choices[i].finish_reason

        print(f"Streaming Finish Reason: {last_finish_reason_list}")

        for i in range(len(last_finish_reason_list)):
            if last_finish_reason_list[i] != "stop":
                full_reasoning_list[i] = "<INCOMPLETE_CONTENT>" + full_reasoning_list[i]
                full_answer_list[i] = "<INCOMPLETE_CONTENT>" + full_answer_list[i]

        reason = full_reasoning_list
        answer = full_answer_list
    else:
        reason = []
        answer = []

        for i in range(len(res.choices)):
            if hasattr(res.choices[i].message, reasoning_chain_key) and getattr(res.choices[i].message, reasoning_chain_key) is not None:
                reason.append(getattr(res.choices[i].message, reasoning_chain_key))
            else:
                reason.append("")

            answer.append(res.choices[i].message.content)

    return reason, answer

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=20), after=after_log(logger, logging.INFO))
def retry_get_openai_response(
    client: openai.Client,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    max_tokens: int = None,
    use_default_generation_config: bool = False,
    temperature: float = 0,
    top_p: float = 1,
    n: int = 1,
    stop: list = [],
    stream: bool = False,
    completion: bool = False,
    **kwargs
) -> ChatCompletion:
    # check whether 'messages' in kwargs
    if 'messages' in kwargs:
        messages = kwargs.pop('messages')
    else: 
        messages = []
        if len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

    try:
        if not completion:
            if not use_default_generation_config:
                ret = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stop=stop,
                    stream=stream,
                    **kwargs
                )
            else:
                print("Using default generation config")
                ret = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    stream=stream,
                )
        else:
            ret = client.completions.create(
                model=model,
                prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                stream=stream,
                **kwargs
            )
        
        if not stream:
            if ret.choices is None:
                raise ValueError(f"No response from OpenAI: {ret}")
    except Exception as e:
        logger.info(f"Error occur: {e}")
        raise e

    return ret

def get_openai_response(*args, **kwargs) -> ChatCompletion:
    res = None
    while res is None:
        try:
            res = retry_get_openai_response(*args, **kwargs)
        except Exception as e:
            logger.info(f"Error occur: {e}")

    return res


def safe_llm_call(client: openai.Client, prompt: str, model: str, max_tokens: int = 8192) -> str:
    """Single-prompt convenience wrapper: send a user prompt, return the
    answer text with any <think>...</think> reasoning stripped."""
    res = get_openai_response(
        client=client,
        model=model,
        user_prompt=prompt,
        max_tokens=max_tokens,
        use_default_generation_config=True,
        stream=False,
    )
    output = process_response(res, stream=False)
    response_text = output[0]
    return extract_answer_from_thinking(response_text)