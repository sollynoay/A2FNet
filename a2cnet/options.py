import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class A2FOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="A2FNet options")

        # PATHS
        self.parser.add_argument("--train_path",
                                 type=str,
                                 help="path to the training data txt",
                                 default=os.path.join(file_dir, "label_train_tank.txt"))
        self.parser.add_argument("--test_path",
                                 type=str,
                                 help="path to the testing data txt",
                                 default=os.path.join(file_dir, "label_test_tank.txt"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="./log/")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="dataset to train on",
                                 default="A2FNet",
                                 )

        # TRAINING options
        #self.parser.add_argument("--checkpoint",
                                 #type=str,
                                 #help="the name of the folder to save the model in",
                                 #default="./checkpoint")
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="FrontDataset",
                                 )

        # OPTIMIZATION options
        self.parser.add_argument("--train_batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--val_batch_size",
                                 type=int,
                                 help="batch size",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-3)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=3)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                help="step size of the scheduler", 
                                default=20)

        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=1)
        self.parser.add_argument("--loss",
                                 type=str,
                                 help="loss function",
                                 default="L1")

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
 

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=100)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
     
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options