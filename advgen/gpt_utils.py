import base64
import openai
import tiktoken
import numpy as np


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError \
            (f"""num_tokens_from_messages() is not presently implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


def query_gpt(openai_key: str, user_query: str or list, system_message: str = None, few_shot_examples=None,
              MODEL="gpt-3.5-turbo"):
    openai.api_key = openai_key
    # MODEL = "gpt-3.5-turbo"  # "gpt-3.5-turbo-0301" # "gpt-4-0314"

    messages = []
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})
    if few_shot_examples is not None:
        for example in few_shot_examples:
            messages.append({"role": "system", "name": "example_user", "content": example[0]})
            messages.append({"role": "system", "name": "example_assistant", "content": example[1]})
    messages.append({"role": "user", "content": user_query})

    if isinstance(user_query, str):
        print(">>" * 20)
        print("messages:", messages)
        print("num_tokens_from_messages:", num_tokens_from_messages(messages))

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response['choices'][0]['message']['content']


def query_gpt_recommendation(openai_key: str, fig_dir="", user_query: str = "", num_recent_scenarios: int = 20, adv_generator=None):
    recommendation_system_message = \
        """
        You are an expert in autonomous vehicle (ego vehicle) performance analysis and training scenario generation. You will help analyze collision incidents and identify weak points in the ego vehicle's design and performance. Based on this analysis, you will generate tailored adversarial training scenarios to improve the ego vehicle's robustness and response. 
        For each recently occurred collision scenario, I will provide a detailed analysis that includes:
        1. The location on the ego vehicle where the collision occurred (e.g., front, behind, left, right, front-left, front-right, behind-left, or behind-right).
        2. The specific type of collision the ego vehicle was involved in, such as rear-end collisions, side-swipe collisions, or other types.
        After analyzing all provided scenarios, you will select the most critical area of focus for future training. This area will be chosen from only one of the following: front, behind, left, right, front-left, front-right, behind-left, or behind-right.
        You should only respond in the format as described below:
        RESPONSE FORMAT: {"Situation": "......","Reasoning": "......","Recommendation": "......","CriticalArea": "......"}
        """
    analysis_system_message = \
        """
        You are an expert traffic accident analyst specializing in analyzing vehicle collisions based on visual data and descriptions. Your task is to carefully assess vehicle positions and collision types with precise reasoning. When given a query involving vehicle collisions, focus on providing accurate and detailed answers regarding:
        1. The relative positions of the vehicles involved in the collision (e.g., in front, behind, left, right, etc.).
        2. The reason for the collision, such as rear-end collisions, side-swipe collisions, or other types.
        Make sure to analyze the scenario based on the image and description provided, and offer clear explanations of your reasoning.
        """
    analysis_user_message = \
        """
        Analyze collisions in image. The red vehicle is driving forward, and the car closest to the red vehicle collided with it. Please analyze: 
        1. Where are the collided vehicle in front, behind, left, right, front-left, front-right, behind-left, or behind-right of the red vehicle? 
        2. The reason for the collision, such as rear-end collision, side-swipe collision, or other types.. 
        You need to give me a precise answer.
        """

    analysis_results = []
    collision_scenarios = adv_generator.collision_figs
    if len(collision_scenarios) > 0:
        for cs_path in collision_scenarios:
            with open(cs_path, "rb") as f:
                img_bytes = f.read()
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            user_query = [
                {"type": "text", "text": analysis_user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
            response = query_gpt(openai_key, user_query=user_query, system_message=analysis_system_message, MODEL="gpt-4o")
            analysis_results.append(response)

        user_query = prepare_recommendation_llm_input(analysis_results)
        response = query_gpt(openai_key, user_query=user_query, system_message=recommendation_system_message, MODEL="gpt-4o")
        print(response)
        return response
    return None


def prepare_recommendation_llm_input(analysis_results: list):
    """
    Prepare the input data for the LLM prompt, focusing on trajectories with collisions.
    :return: Formatted string to be inserted into the LLM prompt
    """
    input_data = "Here are the details of several collision scenarios involving the ego vehicle:\n"
    for i, ar in enumerate(analysis_results):
        input_data += "[Collision {}]: {}\n".format(i + 1, ar)

    input_data += \
        """
        Based on these scenarios, please analyze the weak points in the ego vehicle's design or behavior and provide customized recommendations for what adversarial training scenarios should be generated to improve the vehicle's performance in the future. Make sure to specify the following:
        - What type of collision the ego vehicle should be trained to handle.
        - Where on the ego vehicle the collision should be expected.
        Finally, choose one location from front, behind, left, right, front-left, front-right, behind-left, or behind-right as the most critical area of focus for the ego vehicle. 
        You should only respond in the format as described below:        
        RESPONSE FORMAT: {"Situation": "......","Reasoning": "......","Recommendation": "......","CriticalArea": "......"}
        """
    return input_data


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
