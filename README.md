# Noise-As-Clean

This repo is an unofficial implement of paper ["Noise-As-Clean: Learning Self-supervised Denoising from the Corrupted Image"](https://arxiv.org/abs/1906.06878). The original github repo is [here](https://github.com/csjunxu/Noisy-As-Clean-TIP2020) but very complicated as I think.

The denoising result I get is much lower than that is reported in the paper. That is probably resulted by:
1. I only train the model on single image.
2. I donot use much data argumentation. 

***If you can improve the code, please create pull request, or contact me via xinge.yang@kaust.edu.sa***

## How to run it:
`python nac_single_img.py`

## My thinkings:
NAC provides a novel strategy for single image blind denoising, but I think it is a biased estimation and the proof in the paper is not precise. 