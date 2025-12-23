import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download

def download_assets(repo_id, local_dir):
    """
    下载模型或数据集
    :param repo_id: 仓库ID，如 'Qwen/Qwen2.5-7B'
    :param local_dir: 本地存储路径
    :param repo_type: 'model' 或 'dataset'
    """
    print(f"开始下载: {repo_id} 到 {local_dir}...")
    
    try:
        # 使用 snapshot_download 下载整个仓库
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type='model',
            local_dir_use_symlinks=False,  # 直接复制文件而非创建软连接
            resume_download=True,          # 支持断点续传
            max_workers=8                  # 多线程下载
        )
        print(f"下载完成！路径: {os.path.abspath(local_dir)}")
    except Exception as e:
        print('尝试下载数据集')
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                repo_type='dataset',
                local_dir_use_symlinks=False,  # 直接复制文件而非创建软连接
                resume_download=True,          # 支持断点续传
                max_workers=8                  # 多线程下载
            )
            print(f"下载完成！路径: {os.path.abspath(local_dir)}")
        except Exception as e:
            print(f"下载过程中出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face 资产下载工具")
    parser.add_argument("repo", type=str,  help="仓库ID (e.g., Qwen/Qwen2.5-7B)")
    parser.add_argument("--dest", type=str, default='./models', help="本地保存根路径")
    
    args = parser.parse_args()
    
    # 自动拼接路径：$MODEL/Qwen3-1.7B-FP8
    # 这种方式确保了目录结构的一致性
    final_path = os.path.join(args.dest, args.repo.split('/')[-1])
    
    download_assets(args.repo, final_path)