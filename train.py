import torch
from datasets.face_dataset import FaceDataset
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.colorization_gan import ColorizationGAN
from model.cycle_gan import CycleGAN
from model.generator import Generator
from model.discriminator import Discriminator
from model.optimization import init_weights, define_network

from pathlib import Path

from utils.helpers import fix_seed, compute_nbr_parameters, create_checkpoints_dir
from options.training_opt import TrainOptions

if __name__ == "__main__":

    ## Init 
    options = TrainOptions().parse()

    # fix seeds for reproducibility
    fix_seed(options.seed)

    ## Checkpoints directory
    path = Path.cwd() / "check_points" / options.experiment_name
    create_checkpoints_dir(path)

    hyperparams = Hyperparameters(  show_advance=options.print_freq, 
                                    lr=options.lr, 
                                    batch_size=options.batch_size, 
                                    n_epochs=options.n_epochs,
                                    save_weights=options.save_weights_freq,
                                    lambda_disc=options.lambda_D,
                                    lambda_gen=options.lambda_G,
                                    lambda_pcp=options.lambda_PCP,
                                    lambda_mse=options.lambda_MSE,
                                    num_threads=options.num_threads)

    # dumping hyperparams to keep track of them for later comparison/use
    path_dump_hyperparams = path / "train_options_.txt"
    hyperparams.dump(path_dump_hyperparams)

    ## Dataloader
    dataset = FaceDataset(options, hyperparams, do_transform=True)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, num_workers=hyperparams.num_threads) 

    if options.net_name.upper() == 'COLORIZATION':

        # Generator
        generator = Generator(in_channels=1, out_channels=3, num_features=64, n_res_layers=9, norm_type=options.norm_type, activation='relu')

        # Discriminator
        discriminator = Discriminator(in_channels=3, features=[64, 128, 256, 512], norm_type=options.norm_type)

        print("-----------------------------------------------------")
        print(f"[INFO] Number of trainable parameters for the Generator : {compute_nbr_parameters(generator)}")
        print(f"[INFO] Number of trainable parameters for the Discriminator : {compute_nbr_parameters(discriminator)}")
        print("-----------------------------------------------------")
        print(f"[INFO] Initializing the networks...")
        generator=init_weights(generator, init_type=options.init_type)
        discriminator=init_weights(discriminator, init_type=options.init_type)
        print("-----------------------------------------------------")
        device_ids = [i for i in range(torch.cuda.device_count())]
        print(f"[INFO] Setting up {len(device_ids)} GPU(s) for the networks...")
        print(f"[INFO] .... using GPU(s) device_ids : {device_ids} ...")
        print("-----------------------------------------------------")

        generator=define_network(generator, hyperparams.device, device_ids)
        discriminator=define_network(discriminator, hyperparams.device, device_ids)
        print("-----------------------------------------------------")

        network = ColorizationGAN(generator, discriminator, hyperparams, options)

        ## Start Training
        network.train(dataloader)

    elif options.net_name.upper() == 'CYCLEGAN':

        # Initialize networks (2 Generators and 2 Discriminators)
        network = CycleGAN(hyperparams, options)
        
        ## Start Training
        network.train(dataloader)

    else:
        raise NotImplementedError('Network [%s] is not implemented', options.net_name)