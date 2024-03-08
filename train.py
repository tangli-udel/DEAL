import argparse
parser = argparse.ArgumentParser(description='Concept-level Explanations.')
parser.add_argument('--dataset', default='imagenet', type=str, help='imagenet, cub')
parser.add_argument('--model', default='ViT-B/32', type=str, help='RN50, ViT-B/32')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
parser.add_argument('--gpu_id', default="0", type=str, help='gpu id')
parser.add_argument('--augment_text', action='store_true', help='augment text with concepts.')
parser.add_argument('--save_path', default="./", type=str, help='path to folder saving the checkpoints.')
parser.add_argument('--lambda_sep', default=0.05, type=float, help='Lambda Separation.')
parser.add_argument('--lambda_con', default=0.01, type=float, help='Lambda Consistency.')
parser.add_argument('--lambda_spa', default=0.0005, type=float, help='Lambda Sparsity.')
parser.add_argument('--save_freq', default=5, type=int, help='saving frequency (steps).')
parser.add_argument('--checkpoint', default=None, type=str, help='path to the checkpoint.')
parser.add_argument('--freeze_text', action='store_true', help='freeze clip text encoder or not.')
parser.add_argument('--max_step', default=5000, type=int, help='maximum training steps.')
parser.add_argument('--epochs', default=20, type=int, help='maximum training epochs.')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate.')
parser.add_argument('--eval', action='store_true', help='freeze batch norm.')
args = parser.parse_args()
print(args)


import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from load import *
import torchmetrics
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import numpy as np

from loss import BatchSeparationLoss, BatchConsistencyLoss, SparsityLoss
from explainer import gradCAM, interpret


hparams['seed'] = args.seed
hparams['batch_size'] = args.batch_size
hparams['model_size'] = args.model



seed_everything(hparams['seed'])

bs = hparams['batch_size']
dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

# load model
device = torch.device(hparams['device'])
model, preprocess = clip.load(hparams['model_size'], device=device, jit=False) #Best model use ViT-B/32
if args.checkpoint != None:
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict']) 

if args.eval:
    model.eval()
else:
    model.train()

if args.freeze_text:
    for param in model.transformer.parameters():
        param.requires_grad = False
model = model.to(device)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_sep = BatchSeparationLoss()
loss_con = BatchConsistencyLoss()
loss_spa = SparsityLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98),eps=1e-6, weight_decay=0.001) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

step = 0
for epoch in range(args.epochs):
    if step > args.max_step:
            print("Max step, training terminated!")
            break

    for batch_number, batch in enumerate(tqdm(dataloader)):
        if step > args.max_step:
            print("Max step, training terminated!")
            break

        optimizer.zero_grad()

        if step % args.save_freq == 0:
            step_ctr = 0.0
            step_sep = 0.0
            step_con = 0.0
            step_spa = 0.0
            step_total = 0.0
            count = 0

        images, labels = batch
        texts = np.array(label_to_classname)[labels].tolist()

        tokenized_concepts_list = []
        rich_labels = []
        for i in range(len(texts)):
            concepts = gpt_descriptions[texts[i]][:5]
            concatenated_concepts = ', '.join(concepts)
            label = hparams['label_before_text'] + wordify(texts[i]) + hparams['label_after_text'] + " It may contains " + concatenated_concepts
            rich_labels.append(label)
            
            concepts.insert(0, texts[i])
            tokenized_concepts = clip.tokenize(concepts)
            tokenized_concepts_list.append(tokenized_concepts)

        images = images.to(device)
        if args.augment_text:
            rich_labels = clip.tokenize(rich_labels)
            texts = rich_labels.to(device)
        else:
            texts = clip.tokenize(texts)
            texts = texts.to(device)

        attn_map = []
        if hparams['model_size'] in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64']:
            for k in range(len(images)):
                num_texts = tokenized_concepts_list[k].shape[0]
                repeated_image = images[k].unsqueeze(0).repeat(num_texts, 1, 1, 1)
                heatmap = gradCAM(
                    model.visual,
                    repeated_image,
                    model.encode_text(tokenized_concepts_list[k].to(device)),
                    getattr(model.visual, "layer4")
                )
                attn_map.append(heatmap)
        
        elif hparams['model_size'] in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']:
            for k in range(len(images)):
                R_image = interpret(model=model, image=images[k].unsqueeze(0), texts=tokenized_concepts_list[k].to(device), device=device)
                image_relevance = R_image[0]
                dim = int(image_relevance.numel() ** 0.5)
                R_image = R_image.reshape(-1, dim, dim)
                attn_map.append(R_image)
        
        attn_map_label = [item[0] for item in attn_map]
        attn_map_concepts = [item[1:] for item in attn_map]

        batch_sep = args.lambda_sep * loss_sep(attn_map_concepts)
        batch_con = args.lambda_con * loss_con(attn_map_concepts, attn_map_label)
        batch_spa = args.lambda_spa * loss_spa(attn_map_concepts)
        step_sep += batch_sep.item()
        step_con += batch_con.item()
        step_spa += batch_spa.item()

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        batch_ctr = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        step_ctr += batch_ctr.item()
        
        total_loss = batch_ctr + batch_sep + batch_con + batch_spa
        step_total += total_loss.item()

        total_loss.backward()
        optimizer.step()

        count += 1

        if step % args.save_freq == 0:
            with open(f"{args.save_path}logs/ctr_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_ctr/count) + '\n')
            with open(f"{args.save_path}logs/sep_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_sep/count) + '\n')
            with open(f"{args.save_path}logs/con_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_con/count) + '\n')
            with open(f"{args.save_path}logs/spa_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_spa/count) + '\n')
            with open(f"{args.save_path}logs/total_loss.log", "a") as f:
                f.write(str(step) + ' ' + str(step_total/count) + '\n')

            # print("[CTR LOSS]: " + str(step_ctr/count))
            # print("[SEP LOSS]: " + str(step_sep/count))
            # print("[CON LOSS]: " + str(step_con/count))
            # print("[SPA LOSS]: " + str(step_spa/count))
            # print("[TOTAL LOSS]: " + str(step_total/count) + "\n")

            torch.save({
                'epoch': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f"{args.save_path}step_{step}.pt") #just change to your preferred folder/filename
        
        step += 1