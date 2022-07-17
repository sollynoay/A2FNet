import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import json
import data
import models

from util import sec_to_hm_str

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        if self.opt.no_cuda:
            self.model = models.A2FNet()
        else:
            self.model = models.A2FNet()
            self.model.cuda()

        self.model_optimizer = optim.Adam(self.model.parameters(), self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.model_optimizer, mode='min', factor=0.5, patience =self.opt.scheduler_step_size, verbose = 1)

        if self.opt.load_weights_folder is not None:
            self.load_model()
        
        if self.opt.loss == "L1":
            self.criterion = nn.L1Loss()
        
        datasets_dict = {"FrontDataset": data.FrontDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        transformations = transforms.Compose([transforms.ToTensor()])
        train_filenames = self.opt.train_path
        train_dataset = self.dataset(
            train_filenames, transformations, is_train=True)
        test_filenames = self.opt.test_path
        test_dataset = self.dataset(
            test_filenames, transformations, is_train=False)
        self.train_loader = DataLoader(
            train_dataset, self.opt.train_batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(
            test_dataset, self.opt.val_batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        num_train_samples = len(train_dataset)
        self.num_total_steps = num_train_samples // self.opt.train_batch_size * self.opt.num_epochs
        
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
        self.save_opts()


    def load_model(self):
        """Load model(s) from disk
        """
        model_path = self.opt.load_weights_folder
        self.model.load_state_dict(torch.load(model_path))

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)
    
    def set_train(self):
        """Convert models to training mode
        """
        self.model.train()

    def set_eval(self):
        """Convert models to training mode
        """
        self.model.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        

        print("Training")
        self.set_train()
        count = 0
        _loss = 0
        for i, data in enumerate(self.train_loader):

            before_op_time = time.time()

            img,front,front_hd,ele,mask = data
            img,front,front_hd,ele,mask = img.cuda(),front.cuda(),front_hd.cuda(),ele.cuda(),mask.cuda()
        
            output = self.model(img)

            loss = self.criterion(output*100,front_hd*100)
            _loss = _loss+loss.item()
            self.model_optimizer.zero_grad()
            loss.backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            self.step += 1
            count = count+1
            early_phase = i % self.opt.log_frequency == 0
            if early_phase:
                self.log_time(i, duration, loss.cpu().data)
                self.log("train", img, output, _loss/count)
        self.log_average(_loss/count)
        self.log_epoch("train", _loss/count)
                

            

        count = 0
        _loss = 0
        print("Testing")
        for i, data in enumerate(self.test_loader):
             with torch.no_grad():
                self.model.eval()
                img,front,front_hd,ele,mask = data
                img,front,front_hd,ele,mask = img.cuda(),front.cuda(),front_hd.cuda(),ele.cuda(),mask.cuda()

                output = self.model(img)

                loss = self.criterion(output*100,front_hd*100)
                _loss = _loss+loss.item()
                early_phase = i % self.opt.log_frequency == 0
                if early_phase:
                    self.log("val", img, output, _loss)  
                count = count + 1
        self.log_average(_loss/count)
        self.log_epoch("val", _loss/count)
        self.model_lr_scheduler.step(_loss/count)


    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "checkpoints")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        save_path = save_folder + '/' + str(self.epoch + 1) + '.pth'
        torch.save(self.model.state_dict(), save_path)

    def log(self, mode, inputs, outputs, loss):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        writer.add_scalar('Loss', loss, self.step)
        writer.add_image('Input', inputs[0], self.step)
        writer.add_image('Output', outputs[0], self.step)
        #for l, v in loss.item():
            #writer.add_scalar("{}".format(l), v, self.step)
    def log_epoch(self, mode, loss):
        writer = self.writers[mode]
        writer.add_scalar('Average Loss', loss, self.step)


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.train_batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    def log_average(self, loss):
        print_string = "epoch {:>3} | average loss: {:.5f} | "
        print(print_string.format(self.epoch, loss))
 
    




