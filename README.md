<div id="top" align="center">

<p align="center">
  <a href="https://arxiv.org/abs/2502.15119">
    <img src="https://capsule-render.vercel.app/api?type=soft&height=80&color=timeGradient&text=CurricuVLM:%20Towards%20Safe%20Autonomous%20Driving%20via%20Personalized%20Safety-Critical-nl-Curriculum%20Learning%20with%20Vision-Language%20Models&section=header&fontSize=20" alt="Header Image">
  </a>
</p>

<p align="center">
  <strong>
    <h3 align="center">
      <a href="https://zihaosheng.github.io/CurricuVLM/">Website</a> | 
      <a href="https://arxiv.org/abs/2502.15119">Paper</a> | 
      <a href="https://www.youtube.com/watch?v=esuJEABHVj4">Video</a>  
    </h3>
  </strong>
</p>

</div>


## 📢 News
- **2026.02**: 🚀🚀 **CurricuVLM** is now fully open-sourced! Model weights coming soon!
- **2026.01**: 🔥🔥 **CurricuVLM** has been accepted to *Transportation Research Part C: Emerging Technologies*!


## 😎 Highlights <a name="highlight"></a>
🔥 To the best of our knowledge, **CurricuVLM** is the first work to utilize VLMs for dynamic curriculum generation in closed-loop autonomous driving training.

🏁 **CurricuVLM** outperforms state-of-the-art baselines, across both regular and safety-critical scenarios, achieving superior performance in terms of navigation success, driving efficiency, and safety metrics..

### Before training with CurricuVLM
|                                                       Case 1                                                        |                                                       Case 2                                                        |                                                       Case 3                                                        |                                                       Case 4                                                        |                                                       Case 5                                                        |
|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
| ![Case 1](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case1-adv-combined.gif) | ![Route 2](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case2-adv-combined.gif) | ![Route 3](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case3-adv-combined.gif) | ![Route 4](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case4-adv-combined.gif) | ![Route 5](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case5-adv-combined.gif) |

### After training with CurricuVLM
|                                                       Case 1                                                        |                                            Case 2                                            |                                            Case 3                                            |                                            Case 4                                            |                                           Case 5                                            |
|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| ![Route 6](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case1-trained-combined.gif) | ![Route 7](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case2-trained-combined.gif) | ![Route 8](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case3-trained-combined.gif) | ![Route 9](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case4-trained-combined.gif) | ![Case 5](https://raw.githubusercontent.com/zihaosheng/CurricuVLM/html/static/images/case5-trained-combined.gif) |


## 🛠️ Installation

### 1. Environment Setup

Clone the repository and create a dedicated conda environment:

```bash
git clone https://github.com/zihaosheng/CurricuVLM.git
cd CurricuVLM

conda create -y -n curricuvlm python=3.9
conda activate curricuvlm
PYTHONPATH=. 
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

GPU is required, then install TensorFlow:
```bash
pip install tensorflow==2.11.0
```
Note: You may encounter the following pip dependency ERROR about `protobuf` when installing `TensorFlow`:
```bash
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# wandb 0.26.1 requires protobuf!=5.28.0,!=5.29.0,<8,>4.21.0, but you have protobuf 3.19.6 which is incompatible.
```
If so, install the following to proceed:
```bash
pip install wandb==0.13.11
```
Install PyTorch (CUDA 11.6):
```bash
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 2. Pre-trained Models and Dataset

Download the pre-trained checkpoint and WOMD scene data from the
[release page](https://github.com/zihaosheng/CurricuVLM/releases/tag/v0.0.0).

Organize the directory structure as follows:

```
CurricuVLM/
├── advgen/
│   └── pretrained/
│       └── densetnt.bin
├── raw_scenes_500/
├── LICENSE
└── ...
```


## 🚀 Training

### 1. CurricuVLM

```bash
python gpt_RLtrain.py \
    --seed 123 \
    --mode gpt \
    --save_model \
    --openai_key sk-***
```
> If everything is set up correctly, a MetaDrive simulation window will pop up as follows:
![MetaDrive simulation window](https://private-user-images.githubusercontent.com/48112700/571075499-18ef2ec9-6382-4ce8-87eb-b13e5de1335d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Nzc2NjE1MzIsIm5iZiI6MTc3NzY2MTIzMiwicGF0aCI6Ii80ODExMjcwMC81NzEwNzU0OTktMThlZjJlYzktNjM4Mi00Y2U4LTg3ZWItYjEzZTVkZTEzMzVkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA1MDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNTAxVDE4NDcxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQyMWVkY2Y5YzQzZDI2OTdjYWE5ZjUwMzI1NjVkODg1MWFlY2I2MmNhZmIwNjJkMWZkNDY0NjZkN2YwODhjNTEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.cxl0c4nxrWnOY7rlrXWcPrQT_JmYiE8lPGHmKj05WNM)

### 2. RL Baselines

#### (a) SAC / PPO (SB3)

```bash
python run_baselines/sb3_SACtrain.py --seed 123 --mode replay --save_model
python run_baselines/sb3_PPOtrain.py --seed 123 --mode replay --save_model
```

#### (b) Safe RL (OmniSafe)

Before running safe RL baselines, copy the following:
```yaml
SafeMetaDrive:
  env_cfgs:
  # safe meta drive configurations. More details refer to https://github.com/decisionforce/EGPO
    # max iterations of interactions
    horizon: 1500
    # whether to use random traffic
    random_traffic: False
    # the penalty when crash into other vehicles
    crash_vehicle_penalty: 1.
    # the penalty when crash into other objects
    crash_object_penalty: 0.5
    # the penalty when out of road
    out_of_road_penalty: 1.  # training configurations
```

and append it to the end of:

```
~/miniconda3/envs/curricuvlm/lib/python3.9/site-packages/omnisafe/configs/off-policy/YOUR_ALGO.yaml
```

Then run:

```bash
python run_baselines/omnisafe_SACPID.py --seed 123 --mode replay --save_model
python run_baselines/omnisafe_TD3PID.py --seed 123 --mode replay --save_model
```

### 3. Imitation Learning Baselines

#### (a) Install Dependencies

```bash
pip install imitation==1.0.0 --no-deps
```

Download and extract expert demonstrations from the
[release page](https://github.com/zihaosheng/CurricuVLM/releases/tag/v0.0.0):

```bash
tar -xzvf expert_data.tar.gz
```

#### (b) Collect Expert Demonstrations (Optional)
You can also run the following command to collect your own expert demonstration data:

```bash
python collect_expert_data_set.py
```

#### (c) Training

```bash
python run_baselines/imitation_BC.py   --seed 123 --save_model
python run_baselines/imitation_GAIL.py --seed 123 --save_model
python run_baselines/imitation_AIRL.py --seed 123 --save_model
python run_baselines/imitation_SQIL.py --seed 123 --save_model
```

## 📖 Citation

If you find CurricuVLM useful for your research, please consider giving us a star 🌟 and citing our paper:

```BibTeX
@article{sheng2025curricuvlm,
  title={CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models},
  author={Sheng, Zihao and Huang, Zilin and Qu, Yansong and Leng, Yue and Bhavanam, Sruthi and Chen, Sikai},
  journal={Transportation Research Part C: Emerging Technologies},
  year={2026}
}
```

## 🙏 Acknowledgements

This project builds upon several excellent open-source frameworks: [MetaDrive](https://github.com/metadriverse/metadrive), [CAT](https://github.com/metadriverse/cat), [OmniSafe](https://github.com/PKU-Alignment/omnisafe), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Imitation](https://github.com/HumanCompatibleAI/imitation). We sincerely thank the authors and contributors for making their code publicly available.
