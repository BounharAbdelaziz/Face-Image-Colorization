from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image


class FaceDataset(Dataset):

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    def __init__(self, options, hyperparams, do_transform=True):

      """ load the dataset """

      self.options = options
      self.hyperparams = hyperparams
      self.do_transform = do_transform
      self.img_names = [img_name for img_name in os.listdir(options.img_dir)]

      self.h = options.img_size
      self.w = options.img_size

      if self.do_transform:
        self.transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize((options.img_size, options.img_size)),
                                      transforms.Normalize( (0.5) , (0.5) )
                          ])


      print("------------------------------------------------------------------------------------------")
      print(f'[INFO] Total number of images in the training dataset : {len(self.img_names)}')
      print("------------------------------------------------------------------------------------------")

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    # number of images in the dataset
    def __len__(self):
        return len(self.img_names)
    
    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#

    # open the image
    def open_img(self, full_path, color='RGB'):

      not_opened = True
      while not_opened:
        try:
          image = Image.open(full_path).convert(color)
        except Exception as e:
          print(f'[WARN] An error is occuring with Image.open({full_path}). {e}')
        finally :
          not_opened = False

      return image

    # -----------------------------------------------------------------------------#
    # -----------------------------------------------------------------------------#


    # get an image by index
    def __getitem__(self, idx):

      # Loading the image
      full_path = os.path.join(self.options.img_dir, self.img_names[idx]) 
      
      # From each image we create a pair of gray and RGB images
      image_rgb = self.open_img(full_path, color='RGB')
      image_gray = self.open_img(full_path, color='L')

      if self.do_transform:
        image_rgb = self.transforms(image_rgb).to(self.hyperparams.device)
        image_gray = self.transforms(image_gray).to(self.hyperparams.device)

      return image_rgb, image_gray

    # -----------------------------------------------------------------------------#