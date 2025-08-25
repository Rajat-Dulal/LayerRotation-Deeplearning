# Report Video (Demo Video) available at: [Youtube](www.facebook.com)

#### All my replication and experimentation are available inside my_replication_experimentation folder.

#### MNIST 1st layer feature visualiztion.ipynb  -----> Original Replication 

#### Trial1-CIFAR 1st layer feature_simple.ipynb -----> Novel Model 1 (Preprocessing and training configuration change)

#### Trial2-CCIFAR 1st layer feature-2layerModel.ipynb -----> Novel Model 2 (Preprocessing + Training config + New Model Architecture)

#### Trial3-CCIFAR 1st layer feature-3layerModel.ipynb -----> Novel Model 3 (Preprocessing + Training config + Another New Model Architecture)

# Layer rotation: a surprisingly powerful indicator of generalization in deep networks?
### Steps 
#### For my replication and novel models
1. Install Miniconda (Linux 64-bit)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh


At the end of installation, it will prompt:

Do you wish the installer to initialize Miniconda3?

Type yes.

2. Refresh the Shell
source ~/.bashrc

3. Create and Activate a Conda Environment
conda create --name myenv python=3.6
conda activate myenv

4. Install Requirements
pip install -r requirements.txt

5. Clone the Repository
git clone <repo_link>
cd LayerRotation
cd 'A study of layer rotation in standard training settings'

6. Run the jupyter files

MNIST 1st layer feature visualiztion.ipynb  -----> Original Replication 

Trial1-CIFAR 1st layer feature_simple.ipynb -----> Novel Model 1 (Preprocessing and training configuration change)

Trial2-CCIFAR 1st layer feature-2layerModel.ipynb -----> Novel Model 2 (Preprocessing + Training config + New Model Architecture)

Trial3-CCIFAR 1st layer feature-3layerModel.ipynb -----> Novel Model 3 (Preprocessing + Training config + Another New Model Architecture)

------------------------------------------------------------------------------------------------------------

import_task.py and models.py are used to load the data and the untrained models corresponding to the 5 tasks used in the experiments.

rotation_rate_utils.py contains the code for recording and visualizing layer rotation curves

layca_optimizers.py contains the code to apply Layca on SGD, Adam, RMSprop or Adagrad, and to use layer-wise learning rate multipliers when using SGD.

get_training_utils.py contains utilities to get (optimal) training parameters such as learning rate schedules, learning rate multipliers, stopping criteria and optimizers for training of the five tasks

experiment_utils.py contains utilities such as training curves visualization, one-hot encoding, ...

----------------------------------------------------------------------------------------------------------

#### For other models
1. Install Miniconda (Linux 64-bit)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh


At the end of installation, it will prompt:

Do you wish the installer to initialize Miniconda3?

Type yes.

2. Refresh the Shell
source ~/.bashrc

3. Create and Activate a Conda Environment
conda create --name myenv python=3.6
conda activate myenv

4. Install Requirements
pip install -r requirements.txt

5. Clone the Repository
git clone <repo_link>
cd LayerRotation
cd 'A study of layer rotation in standard training settings'

6. Run the Training Script
python train.py

📦 Using tmux for Persistent Training Sessions
1. Install tmux (if not already installed)
sudo apt update && sudo apt install tmux

2. Start a tmux Session
tmux

3. Start Training Inside tmux
python train.py

4. Detach from tmux Session

Press:

Ctrl + B, then D


This keeps your session running in the background, even if you close the terminal or browser.

5. Reattach to the Session Later
tmux attach

☁️ Downloading Trained Model from Remote VM using Google Cloud SDK
1. Install Google Cloud SDK

Follow instructions at: https://cloud.google.com/sdk/docs/install

2. Authenticate and Set Project
gcloud auth login
gcloud config set project your-project-id

3. Download Files from VM
gcloud compute scp --recurse username@vm-name:/path/to/model /local/path --zone=zone-name


Example:

gcloud compute scp --recurse s225207935@layer-rotation-modeling:/home/s225207935/LayerRotation "C:\Users\rjtdu\Downloads" --zone=us-central1-f
