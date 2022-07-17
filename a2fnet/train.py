from train import Trainer
from options import A2FOptions

options = A2FOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()