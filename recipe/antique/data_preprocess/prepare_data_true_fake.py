import argparse
import os
import datasets
import json
# from PIL import Image
# from verl.utils.hdfs_io import copy, makedirs
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import base64
from collections import Counter


DATA_SOURCE = "true_fake"

SYSTEM_PROMPT = '''你是一个乐于助人的助手。'''

def get_user_prompt(question):
    prompt = '''{}'''.format(question)
    return prompt

def load_json_or_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        print(f"Loaded from {file_path}")
        return data
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def load_image_as_bytes(image_path):
    try:
        with open(image_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Failed to load image bytes: {image_path}, error: {e}")
        return None


def process_data(data):
    question = get_user_prompt('''请根据所给瓷器图片，鉴定瓷器的真伪。请在分析后选择调用工具或者回答，最终鉴定结果请按照<answer>真品</answer>或<answer>赝品</answer>的格式输出。''')
    images = data.get("图片url(用\";\"分隔)", [])
    # check if image exist, if not exist, warning, and delete from list
    for img in images:
        if not os.path.exists(img):
            print(f"Warning: Image path does not exist: {img}")
    images = [img for img in images if os.path.exists(img)]
    ground_truth = data.get("鉴定结果")

    if not question or not images or not ground_truth:
        return None
    # if len(images) != 1:
    #     return None

    # image_path = os.path.join(image_dir, images[0])
    # if not os.path.exists(image_path):
    #     print(f"Image not found: {image_path}")
    #     return None

    question = '<image>\n'*len(images) + question
    processed = {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": get_user_prompt(question)}
        ],
        "images": images,
        "ability": "antique",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": data
    }

    return processed

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", required=True, help="Path to save the processed data")

    args = parser.parse_args()

    output_dir = args.output_dir
    # os.makedirs(output_dir, exist_ok=True)

    # 读取 JSON 数据
    data_json = load_json_or_jsonl(args.data_dir)
    if data_json is None:
        print("No valid data found. Exiting.")
        exit(1)

    # data_json = [data for data in data_json if data['id'].endswith('dynasty')]

    # 构建数据列表
    processed_data = [process_data(data) for data in tqdm(data_json)]
    processed_data = [data for data in processed_data if data is not None]
    print(f"Processed {len(processed_data)} valid entries.")

    counter_list = [len(data['images']) for data in processed_data]
    print(Counter(counter_list))



    #show the first 5 entries for debugging
    for i, entry in enumerate(processed_data[:5]):
        print(f"Entry {i+1}:")
        print(f"  Data Source: {entry['data_source']}")
        print(f"  Prompt: {entry['prompt']}")
        print(f"  Images: {entry['images']}")
        print(f"  Ability: {entry['ability']}")
        # print(f"  Env Name: {entry['env_name']}")
        print(f"  Reward Model: {entry['reward_model']}")
        print(f"  Extra Info: {entry['extra_info']}\n")
    # 构建 datasets.Dataset
    dataset = datasets.Dataset.from_list(processed_data)

    # ✅ 自动识别图像路径并懒加载图像
    dataset = dataset.cast_column("images", datasets.Sequence(datasets.Image()))

    # 保存为 Parquet
    dataset.to_parquet(args.output_dir)