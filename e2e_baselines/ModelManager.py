import torch
import random
import wandb
import torch.nn as nn
import torch.optim as optim

import lightning as L

from torchinfo import summary
from itertools import groupby
from eval_functions import get_metrics
from model.E2E_Score_Unfolding import get_fcn_model, get_rcnn_model, get_cnntrf_model

CONST_MODEL_IMPLEMENTATIONS = {
    "FCN": get_fcn_model,
    "CRNN": get_rcnn_model,
    "CNNT": get_cnntrf_model
}

class LighntingE2EModelUnfolding(L.LightningModule):
    def __init__(self, model, blank_idx, i2w, output_path, learning_rate=1e-4) -> None:
        super(LighntingE2EModelUnfolding, self).__init__()
        self.model = model
        self.loss = nn.CTCLoss(blank=blank_idx)
        self.blank_idx = blank_idx
        self.i2w = i2w
        self.learning_rate = learning_rate
        self.accum_ed = 0
        self.accum_len = 0
        
        self.dec_val_ex = []
        self.gt_val_ex = []
        self.img_val_ex = []
        self.ind_val_ker = []

        self.out_path = output_path
        self.train_dataset = None  # set via set_train_dataset() for name lookups

        self.save_hyperparameters(ignore=['model'])

    def set_train_dataset(self, dataset):
        self.train_dataset = dataset

    def _resolve_sample_names(self, sample_indices):
        if self.train_dataset is not None and hasattr(self.train_dataset, 'get_sample_name'):
            return [self.train_dataset.get_sample_name(int(i)) for i in sample_indices]
        return sample_indices

    def forward(self, input):
        return self.model(input)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Use ExponentialLR instead of ReduceLROnPlateau for reliable checkpoint resumption
        # Decays LR by 0.95 every epoch, similar to patience-based reduction
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=0.98,  # Gentle decay: LR *= 0.98 each epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, train_batch, batch_idx):
        X_tr, Y_tr, L_tr, T_tr, sample_indices = train_batch
        X_tr = X_tr.to(self.device)
        Y_tr = Y_tr.to(self.device)
        if isinstance(L_tr, list):
            L_tr = torch.tensor(L_tr, device=self.device)
        else:
            L_tr = L_tr.to(self.device)
        if isinstance(T_tr, list):
            T_tr = torch.tensor(T_tr, device=self.device)
        else:
            T_tr = T_tr.to(self.device)

        predictions = self.forward(X_tr)

        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            names = self._resolve_sample_names(sample_indices)
            print(f"WARNING: NaN/Inf in predictions at batch {batch_idx}, "
                  f"samples={names}, "
                  f"L_tr={L_tr.tolist()}, T_tr={T_tr.tolist()}, "
                  f"input shape={X_tr.shape}", flush=True)
            return None

        predictions = torch.clamp(predictions, min=-50, max=50)

        loss = self.loss(predictions, Y_tr, L_tr, T_tr)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            names = self._resolve_sample_names(sample_indices)
            print(f"WARNING: NaN/Inf loss at batch {batch_idx}, "
                  f"samples={names}, "
                  f"L_tr={L_tr.tolist()}, T_tr={T_tr.tolist()}, "
                  f"pred range=[{predictions.min():.4f}, {predictions.max():.4f}]", flush=True)
            return None

        # Ensure loss is a scalar tensor
        if loss.dim() > 0:
            loss = loss.mean()

        # Skip gradient update if loss is explosively large (finite but enough to
        # destabilise LSTM weights on the next update). 50 is well above the normal
        # range of ~2-4 but catches runaway batches before they corrupt parameters.
        if loss.item() > 50:
            names = self._resolve_sample_names(sample_indices)
            print(f"WARNING: Explosive loss={loss.item():.2f} at batch {batch_idx}, "
                  f"samples={names}, "
                  f"L_tr={L_tr.tolist()}, T_tr={T_tr.tolist()} — skipping update", flush=True)
            return None

        self.log('loss', loss, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def compute_prediction(self, batch):
        X, Y, _, _, _ = batch
        pred = self.forward(X)
        pred = pred.permute(1,0,2).contiguous()
        pred = pred[0]
        out_best = torch.argmax(pred, dim=1)
        decoded = []
        for j, index in enumerate(out_best):
            # CTC decoding: only skip if same as PREVIOUS token (not if exists anywhere in list!)
            if j == 0 or index.item() != decoded[-1]:
                decoded.append(index.item())
        
        # Filter out blank token and skip tokens not in vocabulary
        decoded = [self.i2w[tok] for tok in decoded if tok != self.blank_idx and tok in self.i2w]
        gt = [self.i2w[int(tok.item())] for tok in Y[0] if int(tok.item()) in self.i2w]

        return decoded, gt

    def validation_step(self, val_batch, batch_idx):
        dec, gt = self.compute_prediction(val_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)

    def on_validation_epoch_end(self):        
        
        cer, ser, ler = get_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_CER', cer)
        self.log('val_SER', ser)
        self.log('val_LER', ler)

        # Clear lists for next validation epoch
        self.gt_val_ex = []
        self.dec_val_ex = []

        return ser

    def test_step(self, test_batch, batch_idx):
        dec, gt = self.compute_prediction(test_batch)
        
        dec = "".join(dec)
        dec = dec.replace("<t>", "\t")
        dec = dec.replace("<b>", "\n")
        dec = dec.replace("<s>", " ")

        gt = "".join(gt)
        gt = gt.replace("<t>", "\t")
        gt = gt.replace("<b>", "\n")
        gt = gt.replace("<s>", " ")


        with open(f"{self.out_path}/hyp/{batch_idx}.krn", "w+") as krnfile:
            krnfile.write(dec)
        
        with open(f"{self.out_path}/gt/{batch_idx}.krn", "w+") as krnfile:
            krnfile.write(gt)

        self.dec_val_ex.append(dec)
        self.gt_val_ex.append(gt)
        self.img_val_ex.append((255.*test_batch[0].squeeze(0)).detach().cpu())
    
    def on_test_epoch_end(self) -> None:
        cer, ser, ler = get_metrics(self.dec_val_ex, self.gt_val_ex)

        self.log('val_CER', cer)
        self.log('val_SER', ser)
        self.log('val_LER', ler)
        
        columns = ['Image', 'PRED', 'GT']
        data = []

        nsamples = len(self.dec_val_ex) if len(self.dec_val_ex) < 5 else 5
        random_indices = random.sample(range(len(self.dec_val_ex)), nsamples)

        for index in random_indices:
            data.append([wandb.Image(self.img_val_ex[index]), "".join(self.dec_val_ex[index]), "".join(self.gt_val_ex[index])])
        
        table = wandb.Table(columns= columns, data=data)
        # Prefer the logger's experiment.log if it exists (wandb via Lightning),
        # otherwise use wandb.log directly. Also try logger.log_table as a fallback.
        try:
            if hasattr(self, 'logger') and getattr(self.logger, 'experiment', None) is not None and hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log({'Test samples': table})
            elif wandb.run is not None:
                wandb.log({'Test samples': table})
            else:
                # Try Lightning logger API
                try:
                    self.logger.log_table(key='Test samples', columns=columns, data=data)
                except Exception:
                    # Best-effort: ignore if no logger supports tables
                    pass
        except Exception:
            # Swallow any logging errors to avoid interrupting training/test
            try:
                wandb.log({'Test samples': table})
            except Exception:
                pass
        self.gt_val_ex = []
        self.dec_val_ex = []
        self.img_val_ex = []

        return ser

def get_model(maxwidth, maxheight, in_channels, out_size, blank_idx, i2w, model_name, output_path, learning_rate=1e-4):
    # Calculate maxlen for transformer-based models (CNNT)
    # After CNN downsampling: height//16 and width//8, then flattened
    # Add 10% safety margin for padding/rounding in CNN
    if model_name in ['CNNT', 'StaveCNNT']:
        maxlen = int(((maxheight // 16) * (maxwidth // 8)) * 1.1)
    else:
        maxlen = None
    
    # Pass maxlen to model constructor
    if maxlen is not None:
        model = CONST_MODEL_IMPLEMENTATIONS[model_name](maxwidth, maxheight, in_channels, out_size, maxlen=maxlen)
    else:
        model = CONST_MODEL_IMPLEMENTATIONS[model_name](maxwidth, maxheight, in_channels, out_size)
    
    lighningModel = LighntingE2EModelUnfolding(model=model, blank_idx=blank_idx, i2w=i2w, output_path=output_path, learning_rate=learning_rate)
    
    try:
        summary(lighningModel, input_size=([1, in_channels, maxheight, maxwidth]))
    except Exception as e:
        print(f"Warning: Could not generate model summary (this is okay): {str(e)[:100]}")
    
    return lighningModel, model