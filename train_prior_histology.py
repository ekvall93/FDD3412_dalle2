""" Contribution: https://github.com/lucidrains/DALLE2-pytorch """

import pickle as pkl
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter, OpenClipAdapter
import wandb

import ipdb
import random


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
    
    captions = tokenizer(captions, return_tensors='pt',  padding='max_length', truncation=True, max_length=77)
  
    return images, captions

dataset = ClipDataset(image_caption_tuple)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)

epoch = 30

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
).cuda()

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 100,
    cond_drop_prob = 0.2
).cuda()



epochs = 20


optim = torch.optim.AdamW(diffusion_prior.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)


cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
with wandb.init(project="train_sd", group="train"):
    for i in range(epochs):
        step = 0
        for b in tqdm(data_loader):
            images, captions = b
            images = images.cuda()
            #captions = {k: v.cuda() for k, v in captions.items()}
            captions = captions['input_ids'].cuda()
            #loss = decoder(images, captions)
            loss = diffusion_prior(captions, images)
            wandb.log({"train_mse": loss.item(),
                            "learning_rate": scheduler.get_last_lr()[0]})
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            if step % 100 == 0:
                diffusion_prior.eval()
                with torch.no_grad():
                    images = images[:30]
                    captions = captions[:30]
                    images_embeddings = clip.embed_image(images)                    
                    captions_embeddings = diffusion_prior.sample(captions)
                    
                    similarity = cosine_similarity(images_embeddings.image_embed, captions_embeddings).mean()
                    wandb.log({"Cosinse similarity": similarity.item()})
                    
                    
                    if step % 1_000 == 0:
                        torch.save(diffusion_prior.state_dict(), f"/data/ekvall/wandb/clip_prior_epoch_{i}_step_{step}_state_dict.pt")
                diffusion_prior.train()
                
            step += 1
        
        
        scheduler.step()
        
        
    #diffusion_prior.save(f"./working/clip_prior_epoch_{i}_final.pt")
    torch.save(diffusion_prior.state_dict(), f"/data/ekvall/wandb/clip_prior_epoch_{i}_final_state_dict.pt")