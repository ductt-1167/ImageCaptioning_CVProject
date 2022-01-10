# Setup
- Firstly, install the list needed packages for project by run command:
  - $pip install -r requirements.txt
- Edit link data in file config_params.py. you can download Flickr-8K dataset in this [link](https://data-flair.training/blogs/python-based-project-image-caption-generator-cnn/)
- Edit hyper-parameters in config_params.py. Note: the first training, config extraction=True to extract feature images and save them with format npy in folder data_npy_folder.
# Directory structure
- Project 
  - checkpoint/
  - data_npy_folder/
  - source files .py

# Training and Inference
- For **Training**, run file training.py (Remember set extraction=True for the first time training, else set False)
- For **Evaluate** model, run file evaluate.py
- For **Predict** caption one image, config image path and run file predict_caption_one_image.py

# Note
- Folder **data_npy_folder** include feature image with npy format, first time training, set extraction=true and training. If test or evaluate, not care this param.
- You can download my checkpoint (10 epochs training) in [here](https://drive.google.com/file/d/1GTbhZ3hbjjc3fU94tcTXPdf_jUr5HcUp/view?usp=sharing) to evaluate or inference model

