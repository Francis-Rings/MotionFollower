# MotionFollower

This repository is the official implementation of **MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion**.

**[MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion](https://arxiv.org/abs/2405.20325)**
<br/>
Shuyuan Tu, [Qi Dai](https://scholar.google.com/citations?user=NSJY12IAAAAJ), Zihao Zhang, Sicheng Xie, [Zhi-Qi Cheng](https://scholar.google.com/citations?user=uB2He2UAAAAJ), [Chong Luo](https://www.microsoft.com/en-us/research/people/cluo/), [Xintong Han](https://xthan.github.io/), [Zuxuan Wu](https://zxwu.azurewebsites.net/), [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=zh-CN)
<br/>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://francis-rings.github.io/MotionFollower/) [![arXiv](https://img.shields.io/badge/arXiv-2405.20325-b31b1b.svg)](https://arxiv.org/abs/2405.20325)

<table class="center" style="width: 100%;">
  
  <tr>
    <td colspan="4" style="text-align:center;">
      <video muted="" autoplay="autoplay" loop="loop" src="https://github.com/Francis-Rings/MotionFollower/assets/12442237/efadf695-d927-49fe-887d-5f93cf8747b4" style="width: 100%; height: auto;"></video>
    </td>
  </tr>
  <tr>
    <td width="25%" style="text-align:center;"><b>&nbsp; Source</b></td>
    <td width="25%" style="text-align:center;"><b>&nbsp; Target</b></td>
    <td width="25%" style="text-align:center;"><b>MotionEditor</b></td>
    <td width="25%" style="text-align:center;"><b>MotionFollower</b></td>
  </tr>
</table>



<p align="center">
<img src="./assets/overview.jpg" width="1080px"/>  
<br>
<em>MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion</em>
</p>


## Abstract
> Despite impressive advancements in diffusion-based video editing models in altering video attributes, there has been limited exploration into modifying motion information while preserving the original protagonist's appearance and background. In this paper, we propose MotionFollower, a lightweight score-guided diffusion model for video motion editing. To introduce conditional controls to the denoising process, MotionFollower leverages two of our proposed lightweight signal controllers, one for poses and the other for appearances, both of which consist of convolution blocks without involving heavy attention calculations. Further, we design a score guidance principle based on a two-branch architecture, including the reconstruction and editing branches, which significantly enhance the modeling capability of texture details and complicated backgrounds. 
Concretely, we enforce several consistency regularizers and losses during the score estimation.
The resulting gradients thus inject appropriate guidance to the intermediate latents, forcing the model to preserve the original background details and protagonists' appearances without interfering with the motion modification.
Experiments demonstrate the competitive motion editing ability of MotionFollower qualitatively and quantitatively. Compared with MotionEditor, the most advanced motion editing model, MotionFollower achieves an approximately 80% reduction in GPU memory while delivering superior motion editing performance and exclusively supporting large camera movements and actions.



## Contact
If you have any suggestions or find our work helpful, feel free to contact us

Email: francisshuyuan@gmail.com

If you find our work useful, please consider citing it:

```
@article{tu2024motionfollower,
  title={MotionFollower: Editing Video Motion via Lightweight Score-Guided Diffusion},
  author={Shuyuan Tu and Qi Dai and Zihao Zhang and Sicheng Xie and Zhi-Qi Cheng and Chong Luo and Xintong Han and Zuxuan Wu and Yu-Gang Jiang},
  journal={arXiv preprint arXiv:2405.20325},
  year={2024}
}
```
