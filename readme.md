## DTA: A Machine Learning Approach for Predicting Protein-Ligand Binding Affinity

#### This repository contains the code and data for a machine learning approach to predict the binding affinity between a protein and a ligand.

### Requirements

    Python 3.10 or higher
    git
    make 4.2.1


## Usage

Clone this repository to your local machine:

    git clone https://github.com/mahmodDAHOL/DTA.git

Navigate to the cloned repository:

    cd DTA

Download the data from the [here](#https://drive.google.com/file/d/1CJIvzSDgZXSgTB5CpCutwShs7Xs2rEkk/view?usp=share_link) and extract it to the data directory, or you can run command 
    
    make install

this command will install all data and will create virtual enviroment with all required dependencies.

To verify that all thing will go well run:

    make test

if all test passed, all things are ready.

To trainig the gnn run:

    make run 

this commad will train 5 folds, and create the following folders:

- logs: describe what are happening during run time
- models: contains 5 folders, one for each fold
- results: contains evaluation results on test data
- runs: tensorboard for visualizeing model architecture and metrics as graphs 

After you got the models, you can run:

    make clean

to remove data and virtual environment.

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or suggestions, please feel free to contact us at mahmodaldahol010@gmail.com. We welcome contributions and feedback from the community.