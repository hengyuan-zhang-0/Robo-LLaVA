# 1. 环境
- cuda11.8 (nvcc -V)
```
git clone https://github.com/hengyuan-zhang-0/Robo-LLaVA.git
cd Robo-LLaVA
```

```
conda create -n llava python=3.10 -y
conda activate llava
#先下载对应cuda的torch
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"

source ~/switch-cuda.sh 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install ninja
pip install flash-attn --no-build-isolation --no-cache-dir
```

# 2. 训练脚本
```
bash scripts/train/finetune_onevision.sh
```
- 注意：lmms-lab/llava-onevision-qwen2-7b-ov 需要从huggingface下载
- BASE_MODEL_PATH 是 ln -s 到当前目录的checkpoints文件夹下
- --video_folder修改为robovqa下载的videos的地址

# 3. Finetune 数据集
- 下载robovqa数据集

`pip install gsutil`

```
gsutil -m cp -r \ "gs://gdm-robovqa/LICENSE.txt" \ "gs://gdm-robovqa/instructions" \ "gs://gdm-robovqa/json" \ "gs://gdm-robovqa/tfrecord" \ "gs://gdm-robovqa/videos" \ .
```

<!-- - 运行 data_generation.py， 将robovqa下载的json输入， 输出地址自定义，可以参考下面的Yaml编写 -->

- 修改ego-robo.yaml
```
datasets:
  - json_path: ./robvqa_test_1000examples.json
    sampling_strategy: all #random:1000
```
