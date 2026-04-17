<h1>Latent Space Skinning (LSS)</h1>
<h2>Latent‑Space Skinning: Learning Compact Representations for Mesh Animations </h2>

<h2>Introduction</h2>
<p>
This repository presents <strong>Latent Space Skinning (LSS)</strong>. The method takes a high‑level representation of an animation sequence and maps it into a <strong>universal latent embedding</strong> that captures the essential motion dynamics in a highly compressed form.
</p>

<p>
The encoder produces a latent vector describing the input animation, while a second latent representation encodes the character’s rest‑pose mesh. 
These two components are fused and passed to the decoder, which reconstructs the full sequence of animated meshes.
</p>

<p>By operating in this shared latent space, the framework enables:</p>
<ul>
  <li><strong>Animation synthesis</strong> from compact latent codes</li>
  <li><strong>Animation compression</strong> with minimal loss of fidelity</li>
  <li><strong>Animation transfer</strong> to new characters during inference</li>
</ul>

<hr>

<h2>Pipeline Overview</h2>
<p>
The full Latent Space Skinning (LSS) workflow consists of the following stages:
</p>

<ul>
  <li><strong>Data Extraction</strong> – Import character FBX files into Blender, run the extraction script, and export raw mesh vertices and rest‑pose data.</li>

  <li><strong>Preprocessing</strong> – Clean, normalize, and pad all mesh sequences; optionally perform canonical remeshing and clustering; generate displacement and auxiliary data.</li>

  <li><strong>Model Architecture Construction</strong> – Build the encoder–decoder network that maps high‑level animation inputs and rest‑pose embeddings into full mesh sequences.</li>

  <li><strong>Training</strong> – Train the model on the prepared dataset, monitor performance, and save checkpoints for later use.</li>

  <li><strong>Evaluation</strong> – Validate reconstruction quality, inspect latent‑space behavior, and compare predicted meshes against ground truth.</li>

  <li><strong>Inference & Output Generation</strong> – Load a trained checkpoint, apply animation transfer to new characters, and export the reconstructed mesh sequences.</li>

  <li><strong>Visualization in Blender</strong> – Import the generated results back into Blender to inspect, render, or further process the animated meshes.</li>
</ul>

<h3> Full Pipeline Flow</h3>
<p>
<strong>Blender FBX Extraction</strong> → 
<strong>Preprocessing & Padding</strong> → 
<strong>Dataset Construction</strong> → 
<strong>Model Architecture</strong> → 
<strong>Training</strong> → 
<strong>Evaluation</strong> → 
<strong>Inference Output</strong> → 
<strong>Blender Visualization</strong>
</p>

<hr>

<h2>How to Use</h2>
<p>
Download the repository and follow the pipeline steps above to run the code locally. 
Ensure that all dependencies are installed and that the dataset is placed in the correct directory structure.
</p>
## Full Pipeline Instructions

This repository includes all necessary scripts and folder templates to run the full Latent Space Skinning (LSS) pipeline locally.  
Follow the steps below to execute the workflow from Blender extraction to model inference.
Full Pipeline Instructions

This repository contains all necessary scripts and folder templates to run the Latent Space Skinning (LSS) pipeline locally. Follow the steps below to execute the complete workflow—from Blender extraction to model inference.

1. Blender Setup & FBX Import
    Open Blender.
    Navigate to the characters/ directory.
    Ensure each character folder (e.g., x_bot, michelle) contains multiple FBX animation files, such as walking.fbx, turning.fbx, or any additional motion sequences intended for training.
    Add all FBX files that you want to include in the pipeline.
2. Run best_LBS.py in Blender
    This script extracts the following for all animation frames: Mesh vertices,Bone matrices
    Output location: characters/<character_name>/animation_data/
3. Run training_ready_static_data.py in Blender. This script extracts the following:
    Character rest pose
    Bone rest pose
    Output location: characters/<character_name>/static_data/
    After completing this step, Blender is no longer required and can be closed.

4. Run processing_padding.py
This script performs the following operations:
    Processes all animation data
    Applies padding to ensure fixed-size tensors for all characters
    Prepares data for dataset creation
    Result: Each character folder will contain one subfolder per animation sequence:
    Animation1/
    Animation2/
    ...
    AnimationN/
    Each subfolder contains training-ready data for a single animation.
   
5. Run Controller.py
This script handles the training workflow, including: Model training Validation Checkpoint saving. When a new best loss is achieved, the script saves: best_model.pth this checkpoint can used for inference. In the controller you can select how many animation you want for training. You can also change the hidden_size, num_layer of Lstm and other hyper parameters. 

6. Inference & Animation Transfer
    Navigate to the inference_scripts/ directory to access scripts for:
     Animation synthesis
     Animation transfer

7. Go open the blender and use the related script to visualize the outputs from the npy files.

<h2> Depentances </h2>
    You can use the requirements.txt in python  ready system with pip install requirements.txt



