import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from einops import rearrange

import numpy as np

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CALIP_PF_Block(nn.Module):
    """Calip parameter-free"""

    def __init__(self, alpha_t, alpha_s, beta_1, beta_2, beta_3):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        self.alpha_t = alpha_t
        self.alpha_s = alpha_s

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
            
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad = False)

    def forward(self, Fs, Ft, Fv):
        # Fs with shape (batch, HW, C), Ft with shape (K, C), Ft is normalized
        # Fv with shape (batch, 1, C) or (batch, C), Fv is also normalized
        Ft = Ft.unsqueeze(0).repeat([Fv.shape[0], 1, 1]) # (batch, K, C)

        with torch.no_grad():
            if len(Fv.shape) == 2:
                Fv = Fv.unsqueeze(1)

            Fs = Fs / Fs.norm(dim = -1, keepdim = True)
            A = Fs @ Ft.permute(0, 2, 1) # (batch, HW, K)

            Fsa = self.softmax(A / self.alpha_s) @ Ft  # (batch, HW, C)
            Fta = self.softmax(A.permute(0, 2, 1) / self.alpha_t) @ Fs # (batch, K, C)
            Fva = F.adaptive_avg_pool1d(Fsa.permute(0, 2, 1), 1).permute(0, 2, 1) + F.adaptive_max_pool1d(Fsa.permute(0, 2, 1), 1).permute(0, 2, 1) # (batch, 1, C), according to paper, we use sum of max and avg pool

            logit_scale = self.logit_scale.exp()

            # for beta_1
            res = self.beta_1 * logit_scale * Fv @ Ft.permute(0, 2, 1)
            # for beta_2
            res += self.beta_2 * logit_scale * Fv @ Fta.permute(0, 2, 1)
            # for beta_3
            res += self.beta_3 * logit_scale * Fva @ Ft.permute(0, 2, 1)


            return res


class CALIP_FS_Block(nn.Module):
    """Calip parameter attention"""
    def __init__(self, beta_1, beta_2, beta_3,
                 input_dim, pre_dim1, pre_dim2):
        super().__init__()

        self.softmax = nn.Softmax(-1)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3

        self.pre_project = nn.Sequential( # 3 layers
            nn.Linear(input_dim, pre_dim1),
            nn.ReLU(inplace = True),
            nn.Linear(pre_dim1, pre_dim2),
            nn.ReLU(inplace = True),
            nn.Linear(pre_dim2, input_dim * 3)
        )

        self.post_project = nn.Sequential( # only one layer
            nn.Linear(input_dim, input_dim)
        )

        self.scale = input_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad = False)

    def forward(self, Fs, Ft, Fv):
        # Fs with shape (batch, HW, C), Ft with shape (K, C), Ft is normalized
        # Fv with shape (batch, 1, C) or (batch, C), Fv is also normalized
        Ft = Ft.unsqueeze(0).repeat([Fv.shape[0], 1, 1]) # (batch, K, C)
        Fs = Fs / Fs.norm(dim = -1, keepdim = True)
        if len(Fv.shape) == 2:
            Fv = Fv.unsqueeze(1)

        out_t = self.pre_project(Ft) # (batch, K, 3 * C)
        out_s = self.pre_project(Fs) # (batch, HW, 3 * C)

        q_t, k_t, v_t = tuple(rearrange(out_t, 'b q (d k) -> k b q d ', k = 3))
        q_s, k_s, v_s = tuple(rearrange(out_s, 'b q (d k) -> k b q d ', k = 3))

        At = self.softmax(self.scale * q_t @ k_s.permute(0, 2, 1)) # (batch, K, HW)
        As = self.softmax(self.scale * q_s @ k_t.permute(0, 2, 1)) # (batch, HW, K)

        Fta = self.post_project(At @ v_s) # (batch, K, C)
        Fsa = self.post_project(As @ v_t) # (batch, HW, C)
        Fva = F.adaptive_avg_pool1d(Fsa.permute(0, 2, 1), 1).permute(0, 2, 1) + F.adaptive_max_pool1d(Fsa.permute(0, 2, 1), 1).permute(0, 2, 1) # (batch, 1, C), according to paper, we use sum of max and avg pool

        logit_scale = self.logit_scale.exp()

        # for beta_1
        res = self.beta_1 * logit_scale * Fv @ Ft.permute(0, 2, 1)
        # for beta_2
        res += self.beta_2 * logit_scale * Fv @ Fta.permute(0, 2, 1)
        # for beta_3
        res += self.beta_3 * logit_scale * Fva @ Ft.permute(0, 2, 1)

        return res


class CustomCLIP(nn.Module):
    def __init__(self, cfg, prompts, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        for _, param in clip_model.named_parameters():
            param.requires_grad = False

    def forward(self, image):
        pass


class CustomCALIP_PF(CustomCLIP):
    def __init__(self, cfg, prompts, clip_model):
        super(CustomCALIP_PF, self).__init__(cfg, prompts, clip_model)
        self.calip_pf = CALIP_PF_Block(cfg.TRAINER.CALIP.ALPHA_T, cfg.TRAINER.CALIP.ALPHA_S, cfg.TRAINER.CALIP.BETA_1, 
                                       cfg.TRAINER.CALIP.BETA_2, cfg.TRAINER.CALIP.BETA_3, 
                                       )

    def forward(self, image):
        image_features, Fs = self.image_encoder(image.type(self.dtype))

        text_features = self.text_features # fixed

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        Ft = text_features

        x = self.calip_pf(Fs, Ft, image_features)

        return x
        

class CustomCALIP_FS(CustomCLIP):
    def __init__(self, cfg, prompts, clip_model):
        super(CustomCALIP_FS, self).__init__(cfg, prompts, clip_model)
        self.calip_fs = CALIP_FS_Block(cfg.TRAINER.CALIP.BETA_1, cfg.TRAINER.CALIP.BETA_2, 
                                       cfg.TRAINER.CALIP.BETA_3,
                                       self.text_features.shape[-1], cfg.TRAINER.CALIP.PRE_DIM1, cfg.TRAINER.CALIP.PRE_DIM2,
                                       )

    def forward(self, image):
        image_features, Fs = self.image_encoder(image.type(self.dtype))

        text_features = self.text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        Ft = text_features

        x = self.calip_fs(Fs, Ft, image_features).squeeze()

        return x


@TRAINER_REGISTRY.register()
class CALIP_PF(TrainerX): # we implement zero shot only 
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)

        if cfg.TRAINER.CALIP.PREC == "fp32":
            # CLIP's default precision is fp16
            clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        self.clip_model = CustomCALIP_PF(cfg, prompts, clip_model)

    def model_inference(self, image):
        logits = self.clip_model(image).squeeze()

        return logits


@TRAINER_REGISTRY.register()
class CALIP_FS(TrainerX): # need to be trained 

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CALIP.PREC in ["fp16", "fp32"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
    
        if cfg.TRAINER.CALIP.PREC == "fp32":
            # CLIP's default precision is fp16
            clip_model.float()

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        print("Building custom CLIP")
        self.model = CustomCALIP_FS(cfg, prompts, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "calip_fs" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.calip_fs, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("calip_fs", self.model.calip_fs, self.optim, self.sched)

        self.scaler = None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.CALIP.PREC
        
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
