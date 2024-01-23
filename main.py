import os 
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from preparing_data import CloudDataset
from torch.utils.data import DataLoader
import numpy as np 
import tifffile as tiff
from PIL import Image as im
import time 
import matplotlib.pyplot as plt
import subprocess

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" 
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 384  
IMAGE_WIDTH = 384  
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r'C:\Users\AFUR\Documents\1_TEST_PROJECTS\finetuning_BERT\data\38-Cloud_training'
VAL_IMG_DIR = r'C:\Users\AFUR\Documents\1_TEST_PROJECTS\finetuning_BERT\data\38-Cloud_test'
RICH_IMG_PATH = r'C:\Users\AFUR\Documents\1_TEST_PROJECTS\finetuning_BERT\data\38-Cloud_training\training_patches_38-cloud_nonempty.csv' # List of files with >80% useful data 

def get_loaders(
    train_dir, 
    val_dir, 
    batch_size, 
    train_transform, 
    val_transform, 
    num_workers=4, 
    pin_memory=True
): 
    train_dataset = CloudDataset(
        os.path.join(train_dir, 'train_red'), 
        os.path.join(train_dir, 'train_green'), 
        os.path.join(train_dir, 'train_blue'), 
        os.path.join(train_dir, 'train_gt'), 
        RICH_IMG_PATH
    ) 

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=False
    )

    vals_dataset = CloudDataset( 
        os.path.join(val_dir, 'train_red'), 
        os.path.join(val_dir, 'train_green'), 
        os.path.join(val_dir, 'train_blue'), 
        os.path.join(val_dir, 'train_gt'), 
        RICH_IMG_PATH
    )
    vals_dataloader = DataLoader(
        vals_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=False
    )

    return train_dataloader, vals_dataloader

def train(loader, model, optimiser, loss_function, scaler): 
    loop = tqdm(loader) 
    print() 
    
    for (data, targets) in loop: # (image, mask) 
        data = data.to(device=DEVICE) 
        targets = targets.float().unsqueeze(1).to(device=DEVICE) 

        # forward 
        with torch.cuda.amp.autocast(): 
            predictions = model(data) 
            loss = loss_function(predictions, targets) 

            print(loss) 
        
        # backward 
        optimiser.zero_grad() 
        scaler.scale(loss).backward() 
        scaler.step(optimiser) 
        scaler.update() 

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    loop = tqdm(loader) 

    with torch.no_grad():
        for x, y in loop:
            x = x.to(device=DEVICE)
            y = y.to(device=DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def main(): 
    '''
        Model training 
    '''
#     train_loader, value_loader = get_loaders(
#     TRAIN_IMG_DIR, 
#     TRAIN_IMG_DIR, 
#     BATCH_SIZE,
#     None,
#     None,
#     NUM_WORKERS,
#     PIN_MEMORY
#     )

#     model = UNET(in_channels=3, out_channels=1).to(DEVICE) 

#     loss_function = nn.BCEWithLogitsLoss() 
#     scaler = torch.cuda.amp.GradScaler() 
#     optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

# #    check_accuracy(value_loader, model, device=DEVICE) 

#     for epoch in range(NUM_EPOCHS): 
#         train(train_loader, model, optimiser, loss_function, scaler)

#         checkpoint = { 
#             "state_dict": model.state_dict(), 
#             "optimiser": "optimiser.state_dict()" 
#         }
#         torch.save(checkpoint, "checkpoint.pth.tar") 

#         # check_accuracy(train_loader, model, device=DEVICE) 
    
    ''' 
        Model testing 
    '''
    process = subprocess.Popen(['bash', '../temperature_logger.sh'])
    # Loading the model 
    device = torch.device('cpu')
    model = UNET(in_channels=3, out_channels=1) 
    checkpoint = torch.load('checkpoint.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # turn into loop for our test
    image_count = 1
    while True:
        start_time = time.time()
        # Loading our test images (R, G, B) 
        red_test = "raspberry_pi_test/red_patch_25_2_by_5_LC08_L1TP_029044_20160720_20170222_01_T1.TIF"
        green_test = "raspberry_pi_test/green_patch_25_2_by_5_LC08_L1TP_029044_20160720_20170222_01_T1.TIF"
        blue_test = "raspberry_pi_test/blue_patch_25_2_by_5_LC08_L1TP_029044_20160720_20170222_01_T1.TIF"
        
        R = (tiff.imread(red_test) * (255 / 2**16)).astype(np.uint8)
        G = (tiff.imread(green_test)  * (255 / 2**16)).astype(np.uint8)
        B = (tiff.imread(blue_test) * (255 / 2**16)).astype(np.uint8)

        rgb_test = np.stack((R, G, B), axis = 0).astype(np.float32) 
        rgb_test_tensor = torch.from_numpy(rgb_test) 
        rgb_test_tensor = rgb_test_tensor.unsqueeze(0) 

        # Feeding our test (as a tensor) into our model 
        prediction = model(rgb_test_tensor) 
        prediction = prediction.squeeze() 

        # Slightly janky step right now, but I am evaluating the models outputs 
        np_pred = prediction.detach().numpy() 
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        
        np_pred = sigmoid(np_pred) 
        np_pred = np.where(np_pred < 0.7, 0, 255) # threshold at 0.7 feels arbitrary tome 
        image = im.fromarray(np_pred.astype(np.uint8), 'L')

        # Lets save our test image and our mask 
        image.save(f'mask{image_count}.png')
        original_image = rgb_test = np.stack((R, G, B), axis = 0).astype(np.uint8)
        original_image = np.ascontiguousarray(original_image.transpose(1, 2, 0))
        img = im.fromarray(original_image, 'RGB')
        img.save(f'test_image{image_count}.png')
        image_count += 1
        seconds = time.time() - start_time
        print(f'image number: {image_count} loaded, processed and saved in {seconds}')
        #time.sleep(5)

if __name__ == "__main__": 
    main()
