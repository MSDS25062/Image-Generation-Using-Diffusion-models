================================================================================
 README - Assignment 5 (Bonus): Image Generation Using Diffusion Models
 Name   : Afzaal
 Roll No: MSDS25062
================================================================================

FOLDER STRUCTURE
----------------
Afzaal_MSDS25062_05/
|-- MSDS25062_05.py          -> Main code (data loader, forward diffusion,
|                                 U-Net denoising model, training, sampling)
|-- MSDS25062_05_allCode.py  -> Combined code of all project files (mandatory file)
|-- test_single_sample.ipynb -> Loads trained model and generates one image
|                                 from pure noise (used during evaluation/viva)
|-- build_notebook.py        -> Helper script that generates test_single_sample.ipynb
|-- Report.pdf               -> Report with results, loss graphs, and samples
|-- Readme.txt               -> This file
|-- saved_models/
|     |-- diffusion_model_final.pth   -> Final trained model weights
|     |-- diffusion_checkpoint.tar    -> Full checkpoint (model + optimizer + losses)

REQUIREMENTS
------------
Python 3.8+
Install dependencies with:

    pip install torch torchvision pillow matplotlib numpy tqdm

(GPU with CUDA is optional but recommended for faster training. The code
automatically uses GPU if available, otherwise falls back to CPU.)

DATASET
-------
This project uses the animal dataset provided for the assignment (15 animal
classes). Place the dataset on your machine in any folder, structured as:

    animal_data/
        Bear/
            img1.jpg
            img2.jpg
            ...
        Cat/
            ...
        Dog/
            ...
        Lion/
            ...
        Tiger/
            ...

(Dataset itself is NOT included in this submission, as instructed.)

HOW TO RUN
----------
Run the main script from the command line, passing the dataset path:

    python MSDS25062_05.py --dataset_path /path/to/animal_data

Optional arguments (all have sensible defaults):

    --classes              List of animal classes to use
                            (default: Bear Cat Dog Lion Tiger)
    --num_images_per_class  Number of images per class to use for training
                            (default: 20)
    --img_size              Image resolution, e.g. 64 (default: 64)
    --batch_size             Training batch size (default: 8)
    --epochs                 Number of training epochs (default: 10)
    --lr                     Learning rate (default: 0.0001)
    --num_steps              Number of diffusion timesteps T (default: 1000)
    --output_dir              Folder to save plots/generated images (default: outputs)
    --save_dir                 Folder to save model checkpoints (default: saved_models)

EXAMPLE
-------
    python MSDS25062_05.py --dataset_path "D:/datasets/animal_data" --epochs 15 --batch_size 16

WHAT THE SCRIPT DOES
---------------------
1. Loads images from the selected animal classes (AnimalDataset class).
2. Runs the forward diffusion process - adds Gaussian noise to images over
   T=1000 timesteps using the closed-form formula (noise is applied via the
   diffusion schedule, never directly to the raw pixel values without it).
3. Trains a U-Net style denoising model to predict the noise added at each
   timestep, using a custom MSE loss between predicted and true noise.
4. Saves a training loss curve (training_loss.png).
5. Runs the reverse diffusion process to generate new images starting from
   pure Gaussian noise (generated_samples.png, generated_samples_grid.png).
6. Saves a visualization of the forward noising process at several steps
   (noise_addition_steps.png) similar to Figure 1 in the assignment.
7. Saves the trained model to saved_models/.

VIEWING/TESTING A SINGLE TRAINED SAMPLE
-----------------------------------------
Open test_single_sample.ipynb in Jupyter Notebook / JupyterLab and run all
cells. It will:
    - Rebuild the model architecture
    - Load the trained weights from saved_models/diffusion_model_final.pth
    - Run the reverse diffusion process to generate one image from noise
    - Display and save the result as test_single_sample_output.png

Make sure test_single_sample.ipynb is run from the project root directory
(Afzaal_MSDS25062_05/) so that it can find saved_models/diffusion_model_final.pth
using the relative path.

