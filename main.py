import os
import torch
import string
import random
import datetime
import numpy as np
from utils.utils import read_config
from torchvision import transforms
import torch

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'


def main(args):
    if(args.seed>=0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("[INFO] Setting SEED: " + str(args.seed))   
    else:
        print("[INFO] Setting SEED: None")

    if(torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")   
    print("[INFO] Device type:", str(device), flush=True)

    config = read_config()
    if args.dataset == "animalkingdom":
        dataset = 'AnimalKingdom'
    elif args.dataset == "ava":
        dataset = 'AVA'
    else:
        dataset = string.capwords(args.dataset)
    path_data = os.path.join(config['path_dataset'], dataset)

#     Here, if we have an arg for animal, update the path_data.
    print("[INFO] Dataset path:", path_data, flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    train_loader = manager.get_train_loader(train_transform)
    print("[INFO] Train size:", str(len(train_loader.dataset)), flush=True)

    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform)
    print("[INFO] Test size:", str(len(val_loader.dataset)), flush=True)

    # criterion or loss
    import torch.nn as nn
    if args.dataset in ['animalkingdom', 'charades', 'hockey', 'volleyball']:
        criterion = nn.BCEWithLogitsLoss()
    elif args.dataset == 'thumos14':
        criterion = nn.CrossEntropyLoss()

    # evaluation metric
    if args.dataset in ['animalkingdom', 'charades']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision'
    elif args.dataset in ['hockey', 'volleyball']:
        from torchmetrics.classification import MultilabelAccuracy
        eval_metric = MultilabelAccuracy(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Accuracy'
    elif args.dataset == 'thumos14':
        from torchmetrics.classification import MulticlassAccuracy
        eval_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
        eval_metric_string = 'Multiclass Accuracy'

    # model
    model_args = (train_loader, val_loader, criterion, eval_metric, class_list, args.test_every, args.distributed, device)
    if args.model == 'convit':
        from models.convit import ConViTExecutor
        executor = ConViTExecutor(*model_args)
    elif args.model == 'query2label':
        from models.query2label import Query2LabelExecutor
        executor = Query2LabelExecutor(*model_args)
    elif args.model == 'query2labelclipinit':
        from models.query2labelclipinit import Query2LabelCLIPInitExecutor
        executor = Query2LabelCLIPInitExecutor(*model_args)
    elif args.model == 'query2labelclip':
        from models.query2labelclip import Query2LabelCLIPExecutor
        executor = Query2LabelCLIPExecutor(*model_args)
        
    elif args.model == 'timesformer':
        from models.timesformer import TimeSformerExecutor
        executor = TimeSformerExecutor(*model_args)
        
        
    elif args.model == 'timesformerclipinit':
        from models.timesformerclipinit import TimeSformerCLIPInitExecutor
        executor = TimeSformerCLIPInitExecutor(*model_args)
    elif args.model == 'timesformerclipinitvideoguide':
        from models.timesformerclipinitvideoguideV2 import TimeSformerCLIPInitVideoGuideExecutor
        executor = TimeSformerCLIPInitVideoGuideExecutor(*model_args)
#         Peter running model
        
        
    elif args.model == 'timesformerresidualclipinit':
        from models.timesformerresidualclipinit import TimeSformerResidualCLIPInitExecutor
        executor = TimeSformerResidualCLIPInitExecutor(*model_args)
    
    elif args.model == 'videomae':
        from models.videomae import VideoMAEExecutor
        executor = VideoMAEExecutor(*model_args)
    elif args.model == 'videomaeclipinit':
        from models.videomaeclipinit import VideoMAECLIPInitExecutor
        executor = VideoMAECLIPInitExecutor(*model_args)
    elif args.model == 'videomaeclipinitvideoguide':
        from models.videomaeclipinitvideoguide import VideoMAECLIPInitVideoGuideExecutor
        executor = VideoMAECLIPInitVideoGuideExecutor(*model_args)

    elif args.model == 'adaptformer':
        from models.adaptformerm import AdaptFormermExecutor
        executor = AdaptFormermExecutor(*model_args)
    elif args.model == 'adaptformerclipinit':
        from models.adaptformerclipinit import AdaptFormerCLIPInitExecutor
        executor = AdaptFormerCLIPInitExecutor(*model_args)
        
        executor.model.to(device)


#     executor.train(args.epoch_start, args.epochs)
#     executor.save(file_path="./checkpoint_"+args.animal+".pth" if args.animal else "./checkpoint.pth")
    from datasets.datasets import AnimalKingdom
    act_dict = {'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4, 'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8, 'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12, 'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17, 'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21, 'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25, 'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28, 'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31, 'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34, 'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39, 'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43, 'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48, 'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52, 'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56, 'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60, 'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64, 'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69, 'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73, 'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77, 'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81, 'Performing allo-grooming': 82, 'Performing allo-preening': 83, 'Performing copulatory mounting': 84, 'Performing sexual exploration': 85, 'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89, 'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94, 'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99, 'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103, 'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107, 'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111, 'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115, 'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119, 'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123, 'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127, 'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131, 'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136, 'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139}
    file_path = "/AnimalAI/Weights/checkpoint_Bird.pth"
#     images = AnimalKingdom('AnimalAI/Output_Video/Input', act_dict, total_length=16, transform=val_transform, mode='predict') 
#  def get_test_transforms():
        
#         input_mean = [0.48145466, 0.4578275, 0.40821073]
#         input_std = [0.26862954, 0.26130258, 0.27577711]
#         input_size = 224
#         scale_size = 256

#         unique = Compose([GroupScale(scale_size),
#                           GroupCenterCrop(input_size)])
#         common = Compose([Stack(roll=False),
#                           ToTorchFormatTensor(div=True),
#                           GroupNormalize(input_mean, input_std)])
#         transforms = Compose([unique, common])
#         return transforms
    # import torchvision.transforms as transforms
    # transform = transforms.ToTensor()
    # from PIL import Image
    # import os

    
    #def load_images(directory):
    #    images = []
    #    for filename in os.listdir(directory):
    #        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): # Check for common image file extensions
    #            img_path = os.path.join(directory, filename)
    #            with Image.open(img_path) as img:
    #                images.append(img.convert('RGB'))
    #    return images

    
    #process_data = transforms(images)
    #process_data = process_data.view((16, -1) + process_data.size()[-2:])
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images if they are not the same size
    transforms.ToTensor(),  # Convert to tensor
    ])

    # Apply the transform to each image in the list
    tensor_list = [transform(image) for image in image_list]

    # Stack all tensors along a new dimension (creating a batch)
    images_tensor = torch.stack(tensor_list, dim=0).to(device)

    images_tensor = images_tensor.unsqueeze(0)  # Adds a batch dimension

    executor.predict(images_tensor, file_path)

#     executor.predict(process_data, file_path)
#     eval = executor.test()
#     print("[INFO] " + eval_metric_string + ": {:.2f}".format(eval * 100), flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="animalkingdom", help="Dataset: volleyball, hockey, charades, ava, animalkingdom")
    parser.add_argument("--model", default="convit", help="Model: convit, query2label")
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=8, type=int, help="Size of the mini-batch")
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of torchvision workers used to load data (default: 8)")
    parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="1", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--test_part", default=6, type=int, help="Test partition for Hockey dataset")
    parser.add_argument("--zero_shot", default=False, type=bool, help="Zero-shot or Fully supervised")
    parser.add_argument("--split", default=1, type=int, help="Split 1: 50:50, Split 2: 75:25")
    parser.add_argument("--train", default=True, type=bool, help="train or test")
    parser.add_argument("--animal", default="", help="Animal subset of data to use.")
    args = parser.parse_args()
    
    main(args)


