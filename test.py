import torch
from datasets.test_dataset import TestDataset
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.generator import Generator

from pathlib import Path

from utils.helpers import fix_seed, compute_nbr_parameters, tensor_to_img, load_model
from options.testing_opt import TestOptions
from PIL import Image
import os

if __name__ == "__main__":

    ## Init 
    options = TestOptions().parse()
    hyperparams = Hyperparameters(batch_size=options.batch_size, num_threads=options.num_threads)

    # fix seeds for reproducibility
    fix_seed(options.seed)


    # dumping hyperparams to keep track of them for later comparison/use
    path_dump_hyperparams = Path(options.results_dir) / "test_options_.txt"
    hyperparams.dump(path_dump_hyperparams)

    if options.net_name.upper() == 'COLORIZATION':

        # Generator
        generator = Generator(in_channels=1, out_channels=3, num_features=64, n_res_layers=9, norm_type=options.norm_type, activation='relu')

        print("-----------------------------------------------------")
        generator = load_model(generator, options.pretrained_path, device=hyperparams.device, mode='test')
        print("-----------------------------------------------------")
        print(f"[INFO] Number of trainable parameters for the Generator : {compute_nbr_parameters(generator)}")
        print("-----------------------------------------------------")

        print(f"generator.alpha= {generator.alpha}")
        print(f"generator.beta= {generator.beta}")
        exit(0)
        # data
        dataset = TestDataset(options, hyperparams, do_transform=True)
        dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True, num_workers=hyperparams.num_threads) 

        for _, data in enumerate(dataloader) : 
            image_gray, img_name = data
            with torch.no_grad():
                img_rgb = generator(image_gray)
            img_rgb = tensor_to_img(img_rgb, save_path=None, size=None, normal=True)

            save_path = os.path.join(options.results_dir, str(img_name[0])) 
            save_img = Image.fromarray(img_rgb)
            save_img.save(save_path)

    else:
        raise NotImplementedError('Network [%s] is not implemented', options.net_name)