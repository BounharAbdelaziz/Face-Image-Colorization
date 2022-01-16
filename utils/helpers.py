import numpy as np
import torch
import os
import random
from model.optimization import init_weights, define_network

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def write_logs_tb(tb_writer_loss, tb_writer_fake, tb_writer_real, img_fake, img_real_gray, img_real, losses, step, epoch, hyperparams, with_print_logs=True, experiment="train_dnn"):

    for k,v in losses.items():
        tb_writer_loss.add_scalar(
            'loss/{}'.format(k), v, global_step=step
        )
        
    tb_writer_fake.add_image(
        'Images/fake', img_fake, global_step=step
    )

    tb_writer_real.add_image(
        "Images/Gray", img_real_gray, global_step=step
    )

    tb_writer_real.add_image(
        "Images/Real", img_real, global_step=step
    )


    if with_print_logs :
        print()
        print(f"Epoch [{epoch}/{hyperparams.n_epochs}] ({experiment})", sep=' ')
        for k,v in losses.items():
            print(f"{k} : [{v:.7f}]", sep=' - ', end=' - ')

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def write_logs_tb_cyclegan(tb_writer_loss, tb_writer_fake, tb_writer_real, images_fakes, images_real, losses, step, epoch, hyperparams, with_print_logs=True, experiment="train_dnn"):

    for k,v in losses.items():
        tb_writer_loss.add_scalar(
            'loss/{}'.format(k), v, global_step=step
        )

    # Adding generated images to tb
    tb_writer_fake.add_image(
        'Fake/RGB', images_fakes[0], global_step=step
    )

    tb_writer_real.add_image(
        "Real/RGB", images_real[0], global_step=step
    )

    tb_writer_fake.add_image(
        'Fake/GRAY', images_fakes[1], global_step=step
    )

    tb_writer_real.add_image(
        "Real/GRAY", images_real[1], global_step=step
    )

    tb_writer_fake.add_image(
        'Fake/cycle_RGB', images_fakes[2], global_step=step
    )

    tb_writer_real.add_image(
        "Fake/cycle_GRAY", images_fakes[3], global_step=step
    )

    if with_print_logs :
        print()
        print(f"Epoch [{epoch}/{hyperparams.n_epochs}] ({experiment})", sep=' ')
        for k,v in losses.items():
            print(f"{k} : [{v:.7f}]", sep=' - ', end=' - ')

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
            
def compute_nbr_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def setup_network(network, hyperparams, type="generator"):
    print("-----------------------------------------------------")
    print(f"[INFO] Number of trainable parameters for the {type} : {compute_nbr_parameters(network)}")
    print("-----------------------------------------------------")

    print(f"[INFO] Initializing the networks...")
    network=init_weights(network, init_type='kaiming')
    print("-----------------------------------------------------")

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(f"[INFO] Setting up {len(device_ids)} GPU(s) for the networks...")
    print(f"[INFO] .... using GPU(s) device_ids : {device_ids} ...")

    network=define_network(network, hyperparams.device, device_ids)
    print("-----------------------------------------------------")

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def load_model(model, PATH, device='cuda', mode='test'):
    print(f'[INFO] Loading model from {PATH} into device: {device} in mode:{mode}.')

    model.load_state_dict(torch.load(PATH))
    model.to(device)

    if mode == 'train':
        model.train()
    else:
        model.eval()

    return model

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def fix_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def create_checkpoints_dir(path):
    try :
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'[INFO] Checkpoint directory already exists')
    else:
        print(f'[INFO] Checkpoint directory has been created')

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print("The value of {} is {}".format(key, value))

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def get_last_lr(optimizer):
    return optimizer.param_groups[0]['lr']

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#