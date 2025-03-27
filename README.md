# CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models

<div class="is-size-5 publication-authors", align="center">
              <!-- Paper authors -->
                <span class="author-block">
                  <a href="https://weichenfan.github.io/Weichen//" target="_blank">Weichen Fan</a><sup>1</sup>,</span>
                  <span class="author-block">
                    <a href="https://www.amberyzheng.com/" target="_blank">Amber Yijia Zheng</a><sup>2</sup>,</span>
                  <span class="author-block">
                  <a href="https://raymond-yeh.com/" target="_blank">Raymond A. Yeh</a><sup>2</sup>,</span>
                  <span class="author-block">
                    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1✉</sup>
                  </span>
                  </div>
<div class="is-size-5 publication-authors", align="center">
                    <span class="author-block">S-Lab, Nanyang Technological University<sup>1</sup> &nbsp;&nbsp;&nbsp;&nbsp; Department of Computer Science, Purdue University <sup>2</sup> </span>
                    <span class="eql-cntrb"><small><br><sup>✉</sup>Corresponding Author.</small></span>
                  </div>

</p>

<div align="center">
                      <a href="https://arxiv.org/abs/2503.18886">Paper</a> | 
                      <a href="https://weichenfan.github.io/webpage-cfg-zero-star/">Project Page</a> |
                      <a href="https://huggingface.co/spaces/weepiess2383/CFG-Zero-Star">Demo</a>
</div>

---

<!-- ![](https://img.shields.io/badge/Vchitect2.0-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/Vchitect-2.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVchitect-2.0&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/DEMO-Vchitect2.0_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/Vchitect-2.0)
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/Vchitect-XL-2B) -->



⚡️ [Huggingface demo](https://huggingface.co/spaces/weepiess2383/CFG-Zero-Star) now supports text-to-image generation with SD3 and SD3.5.

💰 Bonus tip: You can even use pure zero-init (zeroing out the prediction of the first step) as a quick test—if it improves your flow-matching model a lot, it might not be fully trained yet.

## 🔥 Update and News
- [2025.3.27] 🔥 Supported by [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) now!
- [2025.03.26] 📙 Supported by [Wan2.1GP](https://github.com/deepbeepmeep/Wan2GP) now! 
- [2025.03.25] Paper|Demo|Code have been officially released.


## :astonished: Gallery

<table class="center">
<tr>

  <td><img src="assets/repo_teaser.jpg"> </td> 
</tr>
</table>

<table class="center">
<tr>
  <td><img src="assets/1_comparison.gif"> </td>
  <td><img src="assets/3_comparison.gif"> </td>
</tr>

<tr>
  <td><img src="assets/7_comparison.gif"> </td>
  <td><img src="assets/8_comparison.gif"> </td>
  


<tr>
<td><img src="assets/4_comparison.gif"> </td> 
<td><img src="assets/16_comparison.gif"> </td> 
</tr>

</table>


## Installation

### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

  ```bash
  conda create -n CFG_Zero_Star python=3.10
  conda activate CFG_Zero_Star

  #Install pytorch according to your cuda version
  conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

  ```

### 2. Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Local demo
Host a demo on your local machine.
~~~bash
python demo.py
~~~

## Inference
### 1. Wan2.1
**Noted that zero-steps for wan2.1 is set to 1 (first 2 steps, 4% of the total steps).**
#### a. Text-to-Video Generation
Simply run the following command to generate videos in the output folder. Noted that the current version is using Wan-AI/Wan2.1-T2V-14B-Diffusers with the default setting.
~~~bash
python models/wan/video_infer.py
~~~

The results shown below are all generated with this script.
<table class="center">

  <!-- Pair 1 -->
  <tr>
    <td><img src="assets/wan2.1/1322140014_base.gif" style="width:416px; height:auto;"></td>
    <td><img src="assets/wan2.1/1322140014_ours.gif" style="width:416px; height:auto;"></td>
  </tr>
  <tr>
    <td align="center"><b>CFG</b></td>
    <td align="center"><b>CFG-Zero*</b></td>
  </tr>
  <tr>
    <td colspan="2">
      <b>Prompt:</b> "A cat walks on the grass, realistic"<br>
      <b>Seed:</b> 1322140014
    </td>
  </tr>

  <!-- Pair 2 -->
  <tr>
    <td><img src="assets/wan2.1/1306980124_base.gif" style="width:416px; height:auto;"></td>
    <td><img src="assets/wan2.1/1306980124_ours.gif" style="width:416px; height:auto;"></td>
  </tr>
  <tr>
    <td align="center"><b>CFG</b></td>
    <td align="center"><b>CFG-Zero*</b></td>
  </tr>
  <tr>
    <td colspan="2">
      <b>Prompt:</b> "A dynamic interaction between the ocean and a large rock. The rock, with its rough texture and jagged edges, is partially submerged in the water, suggesting it is a natural feature of the coastline. The water around the rock is in motion, with white foam and waves crashing against the rock, indicating the force of the ocean's movement. The background is a vast expanse of the ocean, with small ripples and waves, suggesting a moderate sea state. The overall style of the scene is a realistic depiction of a natural landscape, with a focus on the interplay between the rock and the water."<br>
      <b>Seed:</b> 1306980124
    </td>
  </tr>

  <!-- Pair 3 -->
  <tr>
    <td><img src="assets/wan2.1/1270611998_base.gif" style="width:416px; height:auto;"></td>
    <td><img src="assets/wan2.1/1270611998_ours.gif" style="width:416px; height:auto;"></td>
  </tr>
  <tr>
    <td align="center"><b>CFG</b></td>
    <td align="center"><b>CFG-Zero*</b></td>
  </tr>
  <tr>
    <td colspan="2">
      <b>Prompt:</b> "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."<br>
      <b>Seed:</b> 1270611998
    </td>
  </tr>

  <!-- Pair 4 -->
  <tr>
    <td><img src="assets/wan2.1/158241056_base.gif" style="width:416px; height:auto;"></td>
    <td><img src="assets/wan2.1/158241056_ours.gif" style="width:416px; height:auto;"></td>
  </tr>
  <tr>
    <td align="center"><b>CFG</b></td>
    <td align="center"><b>CFG-Zero*</b></td>
  </tr>
  <tr>
    <td colspan="2">
      <b>Prompt:</b> "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."<br>
      <b>Seed:</b> 2023
    </td>
  </tr>

</table>



## BibTex
```
@misc{fan2025cfgzerostar,
      title={CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models}, 
      author={Weichen Fan and Amber Yijia Zheng and Raymond A. Yeh and Ziwei Liu},
      year={2025},
      eprint={2503.18886},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18886}, 
}
```

## 🔑 License

This code is licensed under Apache-2.0. The framework is fully open for academic research and also allows free commercial usage.


## Disclaimer

We disclaim responsibility for user-generated content. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities. It is prohibited for pornographic, violent and bloody content generation, and to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.
