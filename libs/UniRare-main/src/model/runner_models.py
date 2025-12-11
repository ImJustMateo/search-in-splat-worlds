import time 
import torch
from PIL import Image
from torchvision import transforms


def run_model(args, model,file_opener,img_dir, DEFAULT_DEVICE):
    start_time = time.time()



    saliency, layers = None, None
    if args.model == "Unisal":
        tensor_image = file_opener.open_image(
            file = img_dir, 
        )
        
        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency,layers = model(tensor_image, source="SALICON",get_all_layers= True)

        saliency = torch.exp(saliency)
        saliency = saliency / torch.amax(saliency)
        saliency= saliency.squeeze(0).squeeze(0)
        layers= layers[0]

    if args.model == "TranSalNetDense" or args.model == "TranSalNetRes":

        tensor_image, pad_ = file_opener.open_image(
            file = img_dir, 
        )

        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency, layers = model(tensor_image)

        saliency = saliency.squeeze(0)
        toPIL = transforms.ToPILImage()
        saliency = toPIL(saliency)
        saliency = file_opener.postprocess_img(saliency, img_dir)

        saliency= torch.from_numpy(saliency).type(torch.FloatTensor).unsqueeze(0).to(DEFAULT_DEVICE)

        new_layers= []
        for layer in layers:
            rw= layer.shape[-1] / pad_['w']
            rh= layer.shape[-2] / pad_['h']

            lpad = int(pad_['left'] * rw) +1
            rpad = int(pad_['right'] * rw)+1
            tpad = int(pad_['top'] * rh)+1
            bpad = int(pad_['bottom'] * rh)+1

            layer = layer[:, :, tpad:layer.shape[-2] - bpad, lpad:layer.shape[-1] - rpad]
            new_layers.append(layer)
        layers= new_layers

    if args.model == "TempSal":
        tensor_image = file_opener.open_image(
            file = img_dir, 
        )

        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency, saliency_time, layers = model(tensor_image)
        layers= layers

    # print(f"Processing {args.model}: {time.time() - start_time:.2f} seconds")

    return saliency, layers
