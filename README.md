# Stable Diffusion 3 with added chaos

A Replicate model and ComfyUI workflow to allow for more chaotic and interesting image generation with the same prompt:

https://replicate.com/fofr/sd3-with-chaos

## Running with ComfyUI

You need to load the `workflow_ui.json` workflow.

You'll also need the following custom nodes. You do not need to fix them to specific commits, but these commits are versions that will definitely work:

- [Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) @ `8e51eb3`
- [cubiq/ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials) @ `5f1fc52`
- [WASasquatch/was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui) @ `72290fc`
- [Derfuu/Derfuu_ComfyUI_ModdedNodes](https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes) @ `5c93bd1`

You also need to download your SD3 medium model from HuggingFace and place it in the `models` folder:

https://huggingface.co/stabilityai/stable-diffusion-3-medium

You can read the [Replicate guide to SD3](https://replicate.com/blog/get-the-best-from-stable-diffusion-3) to help decide which safetensors file to use.

## Why

When generating image after image with SD3 I noticed that many of the images all looked very similar. Unless you varied your prompt, you would often see the same colors, lighting and composition.

See this post on X for an example:

https://x.com/fofrAI/status/1803361105613373535

## How it works

While trying to improve the variability of the images I generated, I found that if I provided an init image with a high denoise value then it would loosen up the generation and I would begin to see more variation. I riffed on this idea to make a generic workflow.

### Randomly generate an init image

When creating the init image the workflow will:

- pick a random prompt to determine a base color (either a color, color palette or color combination)
- combine this with a "random, abstract" prompt to get varied coloring and shapes
- append the given image prompt, to maintain colors if they have been prompted for (and to give an initial shape)

This becomes:

```
[color part of prompt], [abstract part of prompt], [given prompt]
```

The final prompt might look something like:

> hazy silver and soft turquoise, argent, 3d structures, random arrangement, abstract art, solid color, a variety of things, structures, solid shapes, very interesting, a landscape photo of a snowy mountain scene

### Generate the init quickly

We do not want to slow down the generation, so the init image is then generated with just 3 steps. Low step outputs from SD3 are already vibrant, but to improve the effect a high CFG is used. It gives a high contrast and colorful init image.

To avoid VAE artefacts from appearing in the final output via image-to-image, the init image is blurred before continuing.

### Control the strength of the effect

This init image is then passed to the main SD3 model, which is run with a low CFG (default of 3.5). The randomness is then varied by changing the denoise strength. At 1.0 it is disabled, values between 0.90 and 0.99 are enough to vary the outputs. Going lower than 0.9 and the image can begin to appear blurry or show VAE artefacts.
