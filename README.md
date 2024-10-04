# Avataar-Assignment_H1
This project demonstrates image generation using **Stable Diffusion** with **ControlNet** for depth conditioning. We generate images based on textual prompts, depth maps, and metadata provided. The objective is to explore the effect of various factors on image generation such as input size, batch size, precision, and aspect ratios.

# Image Generation with ControlNet and Depth Conditioning

## Overview

The assignment has three parts:
1. Generating images based on given metadata and depth.
2. Generating images with varying aspect ratios and analyzing quality.
3. Measuring generation latency and optimizing it, while commenting on quality impact.

## Project Structure

- `/code`: Contains all the Python code files for image generation.
  - `main_code.py`: The main code that generates images based on metadata.
  - `utils.py`: Helper functions for loading images, processing depth maps, etc.
  - `requirements.txt`: Lists all required dependencies.

- `/results`: Contains all the visual results generated during the project.
  - `image1.png`, `image2.png`: Generated images for various prompts and aspect ratios.
  - `test_results.png`: The generated image compared to the reference image (`test_out.png`).

- `README.md`: This file, containing the setup instructions and project description.
- `analysis_report.md`: In-depth analysis of the results, including where the approach works well and where it fails, and answer to the II and III parts of the problem statement.

## Setup Instructions

### Prerequisites

- Python 3.10+
- GPU-enabled environment (recommended)
- Required libraries (listed in `requirements.txt`)

### Note

Due to GPU limitations, the final generated images will be uploaded as soon as the GPU becomes available. The code is fully ready to generate the images, and they will be added to the repository by 5 Oct 2024.

### Installation

Clone this repository:
```bash
https://github.com/nitikasingh21/Avataar-Assignment_H1.git


