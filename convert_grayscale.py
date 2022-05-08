import os
from PIL import Image
from pathlib import Path

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

# open the image
def open_img( full_path, color='RGB'):

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

# saves the image
def save_img(path_saving, image_gray):    
    image_gray.save(path_saving)

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#    
  
if __name__ == "__main__":
  
  PATH_IMG_TO_CONVERT = Path("/home/infres/abounhar/Face-Image-Colorization/test_images/ffhq_val/")
  PATH_SAVE_GRAY_IMG = Path("/home/infres/abounhar/Face-Image-Colorization/test_images/ffhq_val_gray/")
  
  Path(PATH_SAVE_GRAY_IMG).mkdir(parents=True, exist_ok=True)
  
  list_images = os.listdir(PATH_IMG_TO_CONVERT)
  
  for img_name in list_images:
    # convert the image to gray
    image_gray = open_img(PATH_IMG_TO_CONVERT / img_name, color='L')
    # saving the image
    path_saving = os.path.join(str(PATH_SAVE_GRAY_IMG), img_name) 
    save_img(path_saving, image_gray)