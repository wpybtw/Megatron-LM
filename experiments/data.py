import argparse
from datasets import load_dataset, interleave_datasets
import json
import os
from tqdm import tqdm
import os
import json
import shutil
import tempfile
from tqdm import tqdm

def prepare_mixed_dataset(
    output_path, 
    total_tokens_approx, 
    cn_ratio=0.5, 
    buffer_size=10000
):
    # 估算需要的样本总数
    avg_tokens_per_sample = 8192 
    total_samples = int(total_tokens_approx / avg_tokens_per_sample)
    
    print(f"Plan: Generating approx {total_tokens_approx/1e9:.2f}B tokens.")
    print(f"Est. Total Samples: {total_samples}")

    # 1. 加载数据集
    print("Loading FineWeb-Edu (English)...")
    ds_en = load_dataset(
        "HuggingFaceFW/fineweb-edu", 
        name="sample-10BT", 
        split="train", 
        streaming=True
    )
    
    print("Loading SkyPile (Chinese)...")
    ds_cn = load_dataset(
        "Skywork/SkyPile-150B", 
        split="train", 
        streaming=True
    )

    # 2. 统一列名 (修复 NoneType error 的关键步骤)
    # 定义只保留 text 的函数
    def keep_text_only(example):
        return {"text": example["text"]}

    # --- 修复部分开始 ---
    def get_column_names(ds):
        """
        如果 features 为 None，则通过读取第一条数据来获取列名
        """
        if ds.features is not None:
            return list(ds.features.keys())
        else:
            # 这是一个流式数据集，且元数据缺失
            # 我们取第一条数据来看看有哪些 key
            try:
                # 注意：这会消耗一条数据，但对于 TB 级数据可以忽略不计
                # 且 HF dataset 通常支持重新迭代
                sample = next(iter(ds))
                return list(sample.keys())
            except StopIteration:
                return []

    # 获取列名
    cols_en = get_column_names(ds_en)
    cols_cn = get_column_names(ds_cn)
    
    print(f"Columns in EN dataset: {cols_en}")
    print(f"Columns in CN dataset: {cols_cn}")

    # 执行 Map 清洗，移除多余列
    # 注意：interleave 之前必须保证两个数据集 schema 一致（只留 text）
    ds_en = ds_en.map(keep_text_only, remove_columns=[c for c in cols_en if c != 'text'])
    ds_cn = ds_cn.map(keep_text_only, remove_columns=[c for c in cols_cn if c != 'text'])
    # --- 修复部分结束 ---

    # 3. 混合数据集
    mixed_dataset = interleave_datasets(
        [ds_cn, ds_en], 
        probabilities=[cn_ratio, 1-cn_ratio],
        seed=42,
        stopping_strategy="first_exhausted"
    )

    # 4. 写入文件
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    # 假设 mixed_dataset, output_path, total_samples 已经定义

    file_path = os.path.join(output_path, "mixed_corpus.jsonl")
    print(f"Writing to {file_path}...")

    count = 0

    # 使用 NamedTemporaryFile 创建临时文件
    # delete=False 表示关闭文件后不自动删除，我们需要手动移动它
    # mode='w+' 开启读写模式
    # dir='/dev/shm' 可选：如果你确实想强制用内存盘，可以指定 dir='/dev/shm'，否则留空使用系统默认
    # with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8', delete=False, suffix=".jsonl",dir='/dev/shm') as tmp_file:
        # tmp_file_path = tmp_file.name # 获取生成的临时文件路径
    with open(file_path, "w", encoding="utf-8") as f:
        iterator = iter(mixed_dataset)
        try:
            for _ in tqdm(range(total_samples)):
                item = next(iterator)
                
                # 长度过滤
                if len(item['text']) < 100:
                    continue
                # 直接写入目标文件
                json.dump({"text": item["text"]}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                # (可选) 每 100 条强制刷入磁盘，平衡性能与数据安全
                # if count % 10000 == 0:
                #     f.flush()
        except StopIteration:
            print(f"迭代提前结束，共处理 {count} 条数据。")

    # 数据写完后，将临时文件移动到最终目标位置
    # shutil.move 会处理跨文件系统的移动（如果 /dev/shm 和 硬盘 在不同分区）
    # print(f"Moving temporary file from {tmp_file_path} to {file_path}...")
    # shutil.move(tmp_file_path, file_path)

    print(f"Done. Wrote {count} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1e10)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="./corpus_data")
    
    args = parser.parse_args()
    prepare_mixed_dataset(args.out, args.tokens, args.ratio)
