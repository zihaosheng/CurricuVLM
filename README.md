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


> **We are currently in the process of organizing our code and preparing it for release.**
>
> **Stay tuned for our upcoming open-source project on GitHub!**

-----

<br/>

> **[CurricuVLM: Towards Safe Autonomous Driving via Personalized Safety-Critical Curriculum Learning with Vision-Language Models](https://arxiv.org/abs/2502.15119)**
>
> [Zihao Sheng](https://scholar.google.com/citations?user=3T-SILsAAAAJ&hl=en)<sup>1,\*</sup>,
> [Zilin Huang](https://scholar.google.com/citations?user=RgO7ppoAAAAJ&hl=en)<sup>1,\*</sup>,
> [Yansong Qu](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=hIt7KnUAAAAJ)<sup>2</sup>,
> [Yue Leng](https://www.linkedin.com/in/yue-leng-aa8aa363/)<sup>3</sup>,
> [Sruthi Bhavanam](https://www.linkedin.com/in/sruthi-bhavanam-3963489/)<sup>3</sup>,
> [Sikai Chen](https://scholar.google.com/citations?user=DPN2wc4AAAAJ&hl=en)<sup>1,✉</sup><br>
>
> <sup>1</sup>University of Wisconsin-Madison, <sup>2</sup>Purdue University, <sup>3</sup>Google
>
> <sup>\*</sup>Equally Contributing First Authors,
> <sup>✉</sup>Corresponding Author
> <br/>

## 💡 Highlights <a name="highlight"></a>
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
