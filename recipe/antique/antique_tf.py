# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import logging
import os
import random
import re

import requests
# from openai import OpenAI
from PIL import Image

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# openai_api_key = "EMPTY"
# openai_api_base = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.1.100.71:18901/v1")

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# model_name = ""
# if openai_api_base:
#     try:
#         response = requests.get(f"{openai_api_base}/models")
#         response.raise_for_status()
#         models = response.json()
#         if models.get("data"):
#             model_name = models["data"][0]["id"]
#         else:
#             logger.warning("No models found at the specified API base for reward scoring.")
#     except (requests.exceptions.RequestException, KeyError, IndexError) as e:
#         logger.warning(f"Failed to get model from {openai_api_base}: {e}. Reward scoring will be disabled.")


class CustomRLHFDataset(RLHFDataset):
    # prompt_key: 'prompt' by default
    # image_key: 'images' by default
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        row_dict[self.prompt_key] = [
            {
                "role": "system",
                # We don't need tool description, because custom_chat_template will add it.
                "content": (
                    "你是一个乐于助人的助手。你可以调用函数来协助处理用户的请求。"
                    "重要：你一次只能调用一个函数。"
                    "每次调用函数后，如果需要继续调用下一个函数，请先等待该函数的执行结果。"
                ),
            },
            {
                "role": "user",
                "content": row_dict[self.prompt_key][1]["content"],
            },
        ]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                # images = [Image.open(io.BytesIO(image["bytes"])) for image in row_dict.pop(self.image_key)]
                images = [image for image in row_dict.pop(self.image_key)]
                
                # logger.warning(f"Number of images: {len(images)}")

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = {
            "image_zoom_in_tool": {
                "create_kwargs": {"image": images[0]},
                # "execute_kwargs": {},
                # "calc_reward_kwargs": {},
                # "release_kwargs": {},
            }
        }
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


# if "LLM_AS_A_JUDGE_BASE" not in os.environ:
#         try:
#             with open("./tmp/llm_ip.txt") as f:
#                 os.environ["LLM_AS_A_JUDGE_BASE"] = f.read().strip()
#                 print("[Auto Set] LLM_AS_A_JUDGE_BASE from file:", os.environ["LLM_AS_A_JUDGE_BASE"])
#         except Exception as e:
#             print("[Error] Could not load LLM_AS_A_JUDGE_BASE from file:", str(e))

# OPENAI_API_KEY = "EMPTY"
# OPENAI_API_BASE = os.environ.get("LLM_AS_A_JUDGE_BASE", None)
# if not OPENAI_API_BASE:
#     raise ValueError("[DEBUG] LLM_AS_A_JUDGE_BASE environment variable is not set.")
# MODEL_NAME = requests.get(f"{OPENAI_API_BASE}/models").json()['data'][0]['id']
# CLIENT = OpenAI(
#         api_key=OPENAI_API_KEY,
#         base_url=OPENAI_API_BASE,
#     )

import re

def check_model_format(predict_str):
    """
    检查模型回复格式是否符合要求：
    1. 模型回复必须严格按照 <think>...</think><tool_call>...</tool_call> 或 <think>...</think><answer>...</answer>
    2. <think> 内部不得嵌套 <tool_call> 或 <answer>
    3. <tool_call> 和 <answer> 不得同时出现
    4. 只有最后一个模型回复可以出现 <answer>
    5. 每个模型回复必须包含且仅包含一个 <think>...</think>
    6. <think> 必须在 <tool_call> 或 <answer> 之前
    7. 每个模型回复内不允许出现多个 <think>/<tool_call>/<answer>
    8. 所有模型回复中至少要有一个 <answer>
    """
    is_format_error = False
    error_messages = []

    # --- Step 1: 分离系统回复和模型回复 ---
    # system_blocks = re.findall(
    #     r"<\|im_start\|>\s*user\s*<tool_response>.*?</tool_response>\s*<\|im_end\|>",
    #     predict_str, flags=re.DOTALL)
    model_blocks = re.split( r"\s*user\s*<tool_response>.*?</tool_response>\s*", predict_str, flags=re.DOTALL)
    model_blocks = [block.strip() for block in model_blocks if block.strip()]

    # print("System blocks:", len(system_blocks))
    # print("Model blocks:", len(model_blocks))

    total_answer_count = 0  # 所有模型回复的 <answer> 出现次数

    for idx, block in enumerate(model_blocks):
        # think_count = block.count("<think>")
        tool_call_count = block.count("<tool_call>")
        answer_count = block.count("<answer>")
        total_answer_count += answer_count

        # 规则检查
        # if think_count == 0:
            # is_format_error = True
            # error_messages.append(f"模型回复 {idx}: 缺少 <think> 标签")
        # elif think_count > 1:
            # is_format_error = True
            # error_messages.append(f"模型回复 {idx}: 出现了多个 <think> 标签")
        if tool_call_count > 1:
            is_format_error = True
            error_messages.append(f"模型回复 {idx}: 出现了多个 <tool_call> 标签")
        if answer_count > 1:
            is_format_error = True
            error_messages.append(f"模型回复 {idx}: 出现了多个 <answer> 标签")

        # if block.count("<think>") != block.count("</think>"):
            # is_format_error = True
            # error_messages.append(f"模型回复 {idx}: <think> 标签不匹配")

        # think_contents = re.findall(r"<think>(.*?)</think>", block, flags=re.DOTALL)
        # for content in think_contents:
            # if "<tool_call>" in content or "<answer>" in content:
                # is_format_error = True
                # error_messages.append(f"模型回复 {idx}: <think> 内嵌套了非法标签")

        if tool_call_count > 0 and answer_count > 0:
            is_format_error = True
            error_messages.append(f"模型回复 {idx}: 同时存在 <tool_call> 和 <answer>")

        if idx < len(model_blocks) - 1 and answer_count > 0:
            is_format_error = True
            error_messages.append(f"模型回复 {idx}: 非法出现 <answer>（只能在最后一个模型回复出现）")

        # <think> 必须在 <tool_call> 或 <answer> 之前
        # think_pos = block.find("<think>")
        # if tool_call_count > 0 and think_pos > block.find("<tool_call>"):
            # is_format_error = True
            # error_messages.append(f"模型回复 {idx}: <think> 出现在 <tool_call> 之后")
        # if answer_count > 0 and think_pos > block.find("<answer>"):
            # is_format_error = True
            # error_messages.append(f"模型回复 {idx}: <think> 出现在 <answer> 之后")

    # 全局检查
    if total_answer_count == 0:
        is_format_error = True
        error_messages.append("所有模型回复都缺少 <answer> 标签")

    return is_format_error, error_messages



def extract_fields(model_output: str):
    result = {}

    # 所有评分字段
    fields = ["结构", "朝代", "皇帝", "釉色", "纹饰", "器型"]
    for field in fields:
        match = re.search(rf'"?{field}"?\s*:\s*(-?\d+)', model_output)
        result[field] = int(match.group(1)) if match else None

    # # 理由字段提取，允许多行、非贪婪匹配，结尾可无标点
    # match_reason = re.search(
    #   r'"?理由"?\s*:\s*"([\s\S]*?)"\s*$', model_output.strip()
    # )
    # result["理由"] = match_reason.group(1).strip() if match_reason else ""

    return result

# def get_chat_template():
#     chat_template_V1 = """
# ---

# 你是一位中国古陶瓷领域的专家评审员，专门评估模型生成的瓷器鉴定文本质量。

# 请根据以下“参考答案”与“模型输出”，根据如下规则从6个维度进行全面评分，最后只输出对应的评分。

# ---

# ### 🧾 评分要求：

# 请分别从以下 6 个维度，逐项给出 1～5 分评分，要求评分风格偏保守。

# ### 🔍 具体评分规则：

# 1. **器型描述准确性**  
#    是否合理识别并描述了器型；是否与参考答案表述一致，或未表达/表达错误。

# 2. **釉色描述准确性**  
#    是否提及并正确表述釉色特征（如青花、粉彩、红地绿彩等）；是否偏离或缺失。

# 3. **纹饰内容准确性**  
#    是否正确描述装饰图案、题材（如龙凤、花卉、人物、云纹等）；是否与时代风格匹配。

# 4. **朝代判断准确性**  
#    只要模型正确识别出“唐”、“宋”、“元”、“明”、“清”等大朝代，即视为准确，可得满分。无须判断具体皇帝。

# 5. **皇帝判断准确性**
#    （1）如果朝代判断错了，则给0分。
#    （2）判断“参考答案”中是否包含了皇帝名称，如“明嘉靖”或“清雍正”等，不包含的话给-1分（如“清 青花瓷婴戏图碗”）。如果“参考答案”包含了皇帝名称，再判断“模型输出”是否包含了相同的皇帝，如果相同则给5分，如果写错皇帝，给0分。

# ---

# ### ✅ 请严格按照下面的格式返回评分结果，不要返回理由：

# "器型": X,
# "釉色": X,
# "纹饰": X,
# "朝代": X,
# "皇帝": X

# ---

# ### ✅ 示例格式（供模型参考）：

# "器型": 5,
# "釉色": 4,
# "纹饰": 3,
# "朝代": 5,
# "皇帝": -1
# ---
# 以下是参考答案和模型输出：
# [参考答案]：{}
# [模型输出]：{}
# 你的评分是
# """
#     chat_template_V2 = """
# ---

# 你是一位中国古陶瓷领域的专家评审员，专门评估模型生成的瓷器鉴定准确性。

# 请根据以下“参考答案”与“模型输出”，按照5个维度给分，每个维度评分结果控制在0-1之间。

# ---

# ### 🧾 评分要求：

# 1. **鉴定结果结构性**
#     是否严格按照“朝代皇帝 釉色纹饰器型”或者“朝代 釉色纹饰器型”的格式给出瓷器的鉴定结果。
#     如果格式正确，给1分；如果格式不正确，给0分。

# 2. **朝代判断准确性**  
#    只要模型正确识别出“唐”、“宋”、“元”、“明”、“清”等大朝代，即视为准确，可得满分。无须判断具体皇帝。
#    如果模型输出的朝代与参考答案中的朝代一致，给1分；如果不一致，给0分。

# 3. **皇帝判断准确性**
#    （1）如果参考答案中只有朝代没有写具体皇帝年号，给-1。
#    （2）如果“参考答案”即包含朝代，又包含具体皇帝年号，如“明嘉靖”、“清康熙”等，则判断模型是否输出正确的朝代以及皇帝年号：
#       （1.1）若没有输出正确的朝代或者皇帝年号，得0分；
#       （1.2）若正确输出了朝代以及皇帝年号，得1分。

# 4. **釉色描述准确性**  
#    是否提及并正确表述釉色特征（如青花、粉彩、红地绿彩等）；是否偏离或缺失。
#     根据准确程度在0-1之间打分。

# 5. **纹饰内容准确性**  
#    是否正确描述装饰图案、题材（如龙凤、花卉、人物、云纹等）；是否与时代风格匹配。
#     根据准确程度在0-1之间打分。

# 6. **器型描述准确性**  
#    是否合理识别并描述了器型；是否与参考答案表述一致，或未表达/表达错误。
#    根据准确程度在0-1之间打分。





# ### ✅ 请严格按照下面的格式返回评分结果，不要返回理由：

# "结构": X,
# "朝代": X,
# "皇帝": X,
# "釉色": X,
# "纹饰": X,
# "器型": X,


# ---

# ### ✅ 示例格式（供模型参考）：

# "结构": 1,
# "朝代": 1,
# "皇帝": -1,
# "釉色": 0.8,
# "纹饰": 0.6,
# "器型": 1,

# ---
# 以下是参考答案和模型输出：
# [参考答案]：{}
# [模型输出]：{}
# 你的评分是
# """

#     return chat_template_V2


# def get_prompt(ground_truth, output):
#     prompt = get_chat_template().format(ground_truth, output)
#     return prompt


def extract_answer(text):
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    # 使用非贪婪模式匹配<answer>和</answer>之间的内容
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# def compute_score_V1(predict_str: str, ground_truth: str, extra_info=None) -> float:
#     """
#     Compute the score based on the predicted string and ground truth.

#     In this version, the scoring is adjusted to give more weight to the tool usage and format correctness.
    
#     Args:
#         predict_str (str): The predicted string.
#         ground_truth (str): The ground truth string.
#         extra_info (dict, optional): Additional information, not used in this function.

#     Returns:
#         float: The computed score.
#     """
#     # format_score
#     is_format_error = False
#     count_think_1 = predict_str.count("<think>")
#     count_think_2 = predict_str.count("</think>")
#     if count_think_1 != count_think_2:
#         is_format_error = True

#     count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
#     count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
#     if count_vision_1 != count_vision_2:
#         is_format_error = True

#     predict_no_think = predict_str.split('</think>')[-1].strip()
#     count_answer_1 = predict_no_think.count("<answer>")
#     count_answer_2 = predict_no_think.count("</answer>")
#     if count_answer_1 != count_answer_2:
#         is_format_error = True
    

#     answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
#     if len(answer_text) > 5:
#        acc_reward = 0.0
#     elif ground_truth in answer_text:
#         acc_reward = 1.0
#     else:
#         acc_reward = 0.0

#     tool_reward_1 = 1.0 if count_vision_1 > 0 else 0.0
#     tool_reward_2 = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
#     format_reward = -1.0 if is_format_error else 0.0

#     res = 0.8 * acc_reward + 0.2 * format_reward + 0.5 * tool_reward_1 + 0.5 * tool_reward_2

#     return res


# def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
#     """
#     Compute the score based on the predicted string and ground truth.

#     In this version, the scoring is adjusted to give more weight to the tool usage and format correctness.
    
#     Args:
#         predict_str (str): The predicted string.
#         ground_truth (str): The ground truth string.
#         extra_info (dict, optional): Additional information, not used in this function.

#     Returns:
#         float: The computed score.
#     """


#     # format_score
#     is_format_error, error_messages = check_model_format(solution_str)
#     has_tool_usage = bool(
#         re.search(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
#         or re.search(r"<tool_response>.*?</tool_response>", solution_str, re.DOTALL)
#     )
#     if has_tool_usage:
#         is_call_tool = True
#     else:
#         is_call_tool = False
#     # accuracy_score
#     answer_text = solution_str.split("<answer>")[-1].split("</answer>")[0].strip()
#     prompt = get_prompt(ground_truth, answer_text)
#     chat_response = CLIENT.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "你是一个乐于助的评分助手，负责根据模型输出的文本进行评分。"},
#             {"role": "user", "content": prompt},
#         ],
#         seed = random.randint(0, 1000000),
#         temperature=0.3,
#     )
#     response = chat_response.choices[0].message.content.strip()
#     result = extract_fields(response)
#     dct_score = {
#         "结构": result.get("结构", None),
#         "朝代": result.get("朝代", None),
#         "皇帝": result.get("皇帝", None),
#         "釉色": result.get("釉色", None),
#         "纹饰": result.get("纹饰", None),
#         "器型": result.get("器型", None),
#     }
#     acc_reward = 0
#     n_scores = 0
#     for k, v in dct_score.items():
#         if  (v is None) or (v < 0):
#             continue
#         n_scores += 1
#         acc_reward += v
#     if n_scores > 0:
#         acc_reward = acc_reward/n_scores
#     else:
#         acc_reward = 0.0

#     format_reward = -1.0 if is_format_error else 0.0
#     tool_reward_1 = 1.0 if is_call_tool else 0.0
#     tool_reward_2 = acc_reward if is_call_tool  else 0.0

#     res = 0.8 * acc_reward + 0.2 * format_reward + 0.5 * tool_reward_1 + 0.5 * tool_reward_2
#     # Debugging output
#     logger.info(f'''
# [DEBUG]=========================================== 
# Ground Truth: {ground_truth},
# Model Response: {solution_str},
# Extracted JUDGE Result: {result},
# Final Computed Score: {res}, 
# acc_reward: {acc_reward}, 
# format_reward: {format_reward}, 
# tool_reward_1: {tool_reward_1},
# Format Error: {error_messages}
# ===========================================
# ''')
    


#     dct_return = {
#         "score": res,
#         "acc_reward": acc_reward,
#         "format_reward": format_reward,
#         "tool_reward_1": tool_reward_1,
#         "tool_reward_2": tool_reward_2,
#     }
#     return dct_return



def compute_score_tf(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    Compute the score based on the predicted string and ground truth.

    In this version, the scoring is adjusted to give more weight to the tool usage and format correctness.
    
    Args:
        predict_str (str): The predicted string.
        ground_truth (str): The ground truth string.
        extra_info (dict, optional): Additional information, not used in this function.

    Returns:
        float: The computed score.
    """


    # format_score
    is_format_error, error_messages = check_model_format(solution_str)
    has_success_response = bool(
    re.search(r"<tool_response>.*?成功.*?</tool_response>", solution_str, re.DOTALL)
    )

    if has_success_response:
        is_call_tool = True
    else:
        is_call_tool = False
    # accuracy_score
    answer_text = solution_str.split("<answer>")[-1].split("</answer>")[0].strip()
    acc_reward = 0.0
    if "真" in answer_text and "赝" not in answer_text:
        if "真" in ground_truth:
            acc_reward = 1.0
    if "赝" in answer_text and "真" not in answer_text:
        if "赝" in ground_truth:
            acc_reward = 1.0
    # if answer_text == ground_truth:
    #     acc_reward = 1.0
    # else:
    #     acc_reward = 0.0

    format_reward = -1.0 if is_format_error else 0.0
    tool_reward_1 = 1.0 if is_call_tool else 0.0
    tool_reward_2 = acc_reward if is_call_tool  else 0.0

    res = 1.2 * acc_reward + 0.2 * format_reward + 0.5 * tool_reward_1 + 0.5 * tool_reward_2
    # Debugging output
    print(f'''
[DEBUG]=========================================== 
Ground Truth: {ground_truth},
Model Response: {solution_str},
Final Computed Score: {res}, 
acc_reward: {acc_reward}, 
format_reward: {format_reward}, 
tool_reward_1: {tool_reward_1},
Format Error: {error_messages}
===========================================
''')

    dct_return = {
        "score": res,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
        "tool_reward_1": tool_reward_1,
        "tool_reward_2": tool_reward_2,
    }
    return dct_return

