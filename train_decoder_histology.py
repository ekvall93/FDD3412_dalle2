""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import pickle as pkl
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import torch
from dalle2 import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter, OpenClipAdapter
import wandb


clip = OpenClipAdapter(name='hf-hub:wisdomik/QuiltNet-B-32', pretrained=None)

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

image_caption_tuple = pkl.load(open("/data/ekvall/1mquilt/image_caption_tuple.pkl", "rb"))

class ClipDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        # Define the image transformation here

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, caption = self.data_list[idx]
        image = Image.open(image_path).convert("RGB")
        return image, caption

def collate_fn(batch):
    images, captions = zip(*batch)
    images = processor.preprocess(images, return_tensors='pt')['pixel_values']
    
    captions = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
  
    return images, captions

dataset = ClipDataset(image_caption_tuple)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)

epoch = 30


unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    text_embed_dim = 512,
    #cond_on_text_encodings = True  # set to True for any unets that need to be conditioned on text encodings (ex. first unet in cascade)
).cuda()

decoder = Decoder(
    unet = unet1,
    image_sizes = [224],
    clip = clip,
    timesteps = 1000,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
).cuda()


epochs = 20


optim = torch.optim.AdamW(decoder.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
import ipdb
import random


with wandb.init(project="train_sd", group="train"):
    for i in range(epochs):
        step = 0
        for b in tqdm(data_loader):
            images, captions = b
            images = images.cuda()
            #captions = {k: v.cuda() for k, v in captions.items()}
            captions = captions['input_ids'].cuda()
            #loss = decoder(images, captions)
            loss = decoder(images)
            wandb.log({"train_mse": loss.item(),
                            "learning_rate": scheduler.get_last_lr()[0]})
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            if step % 1_000 == 0:
                decoder.eval()
                with torch.no_grad():
                    random.shuffle(images)
                    images = images[:30]
                    images_embeddings = clip.embed_image(images)                    
                    sampled_images = decoder.sample(image_embed=images_embeddings.image_embed)
                    
                    wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})
                    wandb.log({"real_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in images]})
                    #decoder.save(f"./working/clip_decoder_epoch_{i}_step_{step}.pt")
                    torch.save(decoder.state_dict(), f"/data/ekvall/wandb/clip_decoder_epoch_{i}_step_{step}_state_dict.pt")
                decoder.train()
                
            step += 1
        
        
        scheduler.step()
        
        
    #decoder.save(f"./working/clip_decoder_epoch_{i}_final.pt")
    torch.save(decoder.state_dict(), f"/data/ekvall/wandb/clip_decoder_epoch_{i}_final_state_dict.pt")