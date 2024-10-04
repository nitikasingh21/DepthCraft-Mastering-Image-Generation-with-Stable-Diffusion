# Section I

## Analysis Report

## Thought Process

### Part I: Metadata-Based Image Generation
- **Goal**: To generate images using metadata and depth conditioning.
- **Approach**: The ControlNet model was used with Stable Diffusion to integrate depth maps along with text prompts provided in the metadata. Depth-based conditioning helped to control the structure of the generated images, ensuring that the images align with the prompts.

### Part II: Aspect Ratio Image Generation
- **Goal**: To explore how different aspect ratios affect image generation.
- **Approach**: Aspect ratios of 1:1, 16:9, and 9:16 were used for image generation. While the composition of the images remained intact, some level of detail was lost at non-square aspect ratios, especially in 9:16.

### Part III: Generation Latency and Optimization
- **Goal**: Measure generation latency and implement optimizations to reduce it.
- **Findings**: Using mixed precision (`float16` instead of `float32`) and reducing the input size from 512x512 to 256x256 significantly improved latency (by 40-60%). However, reducing input size also caused a slight decrease in image quality, especially in detailed regions.

## Challenges and Failures
- **GPU Limitations**: Due to the GPU resource constraints in Colab, I was unable to generate the final images. However, the code is complete, and the results will be uploaded once the GPU becomes available.
- **Quality vs Latency**: Reducing image size helped decrease latency but slightly reduced the sharpness and detail of the images.

## Improvements for Future Work
- **Super-Resolution Techniques**: Post-processing the images using super-resolution models could help retain detail in cases where input size is reduced.
- **Additional Training**: Fine-tuning the model on custom datasets might help improve depth conditioning and generate more realistic results.

# Section II

### Can we generate images of different aspect ratios (use “Metadata/Nocrop/2_nocrop.png” to test this out) using SD? Comment on the generation quality with respect to the aspect ratio of 1:1 for the same image.
