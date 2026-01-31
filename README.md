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
- **2026.01**: 🔥🔥 **CurricuVLM** has been accepted to *Transportation Research Part C: Emerging Technologies*! We will release the code soon!


## 😮 Highlights <a name="highlight"></a>
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

## 🛠️ Getting Started <a name="setup"></a>


Create a conda env and install the requirements:
```shell
# Clone the repo
git clone https://github.com/zihaosheng/CurricuVLM.git
cd CurricuVLM

# Create a conda env
conda create -y -n curricuvlm python=3.9
conda activate curricuvlm

# Install PyTorch
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Install the requirements
pip install -r requirements.txt
```

## Training Baselines
### RL baselines
SAC/PPO
```bash
python sb3_SACtrain.py --seed=123 --mode=replay --save_model
python sb3_PPOtrain.py --seed=123 --mode=replay --save_model
```
For safe RL, first copy `env_cfgs` from https://github.com/PKU-Alignment/omnisafe/blob/15603dd7a654a991d0a4648216b69d60b81a6366/omnisafe/configs/off-policy/SACLag.yaml#L276 and add to the config files in  `~/miniconda3/envs/curricuvlm/lib/python3.9/site-packages/omnisafe/configs/off-policy/YOUR_ALGO.yaml`
```bash
python omnisafe_SACPID.py --seed=123 --mode=replay --save_model
python omnisafe_TD3PID.py --seed=123 --mode=replay --save_model
```
### IL baselines

## 🎯 Citation <a name="citation"></a>

If you find CurricuVLM useful for your research, please consider giving us a star 🌟 and citing our paper:

```BibTeX
@article{sheng2025curricuvlm,
  title={CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models},
  author={Sheng, Zihao and Huang, Zilin and Qu, Yansong and Leng, Yue and Bhavanam, Sruthi and Chen, Sikai},
  journal={arXiv preprint arXiv:2502.15119},
  year={2025}
}
```
