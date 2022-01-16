
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class BaseModel(nn.Module):

    def __init__(self, experiment="train_dnn") -> None:
        super(BaseModel, self).__init__()
        # Tensorboard logs
        self.experiment = experiment
        self.tb_writer_loss = SummaryWriter(f"logs/{self.experiment}/loss_train_{self.experiment}")