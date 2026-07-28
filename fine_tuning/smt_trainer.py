"""
SMT fine-tuning trainer (PyTorch Lightning module).

Wraps the pre-trained SMT model for fine-tuning with:
  - Configurable learning rate and optimizer
  - Autoregressive validation with SER/CER/LER metrics
  - Gradient clipping
"""

import logging
import torch
import lightning as L
from eval_functions import compute_poliphony_metrics
from tedn_eval import compute_tedn_metrics

logger = logging.getLogger(__name__)


class SMTTrainer(L.LightningModule):

    def __init__(self, model, lr=1e-4, weight_decay=1e-4, label_smoothing=0.1,
                 freeze_encoder_epochs=0, warmup_steps=0, compute_tedn=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        # Fine-tuning a full pretrained encoder+decoder on ~280 samples/fold with no
        # warmup and no freezing is a catastrophic-forgetting/overfitting setup (see
        # FINETUNING_REVIEW.md 2.2/2.3) — these two default to off (0) so existing configs
        # keep their old behavior unless a config opts in.
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.warmup_steps = warmup_steps
        self.compute_tedn = compute_tedn
        self.val_predictions = []
        self.val_ground_truths = []

    def on_train_start(self):
        # from_pretrained() loads models in eval mode; force all submodules to train mode
        self.model.train()
        train_count = sum(1 for m in self.model.modules() if m.training)
        eval_count = sum(1 for m in self.model.modules() if not m.training)
        logger.info(f"Model set to train mode: {train_count} modules in train, {eval_count} in eval")

        # Override model loss with label smoothing to prevent overconfident predictions
        if self.label_smoothing > 0:
            padding_token = self.model.config.padding_token if hasattr(self.model.config, 'padding_token') else 0
            self.model.loss = torch.nn.CrossEntropyLoss(
                ignore_index=padding_token,
                label_smoothing=self.label_smoothing,
            )
            logger.info(f"Applied label_smoothing={self.label_smoothing} to loss function")

        if self.freeze_encoder_epochs > 0:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info(f"Encoder frozen for the first {self.freeze_encoder_epochs} epochs")

    def on_train_epoch_start(self):
        if self.freeze_encoder_epochs > 0 and self.current_epoch == self.freeze_encoder_epochs:
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            logger.info(f"Encoder unfrozen at epoch {self.current_epoch}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if not self.warmup_steps:
            return optimizer

        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def training_step(self, batch, batch_idx):
        x = batch['encoder_input']
        di = batch['decoder_input']
        y = batch['labels']

        output = self.model(encoder_input=x, decoder_input=di, labels=y)
        loss = output.loss

        # NaN-skip guard (2026-07-28): a single degenerate sample can produce a non-finite loss;
        # in fp32 (no AMP grad-scaler to skip it) that NaN propagates through the backward pass and
        # permanently poisons the fold's weights. Returning None skips the optimizer step for this
        # batch so one bad sample can't corrupt the whole run.
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at epoch {self.current_epoch} batch {batch_idx} "
                           f"— skipping this batch (no optimizer step)")
            return None

        bs = x.shape[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)

        # Log every 20 steps so progress is visible in SLURM logs
        if batch_idx % 20 == 0:
            logger.info(f"Epoch {self.current_epoch} | Step {batch_idx}/{self.trainer.num_training_batches} | loss={loss.item():.4f}")

        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get('train_loss_epoch')
        lr = self.optimizers().param_groups[0]['lr']
        if avg_loss is not None:
            logger.info(f"Epoch {self.current_epoch} finished — avg train_loss: {avg_loss:.4f}, lr: {lr:.2e}")
        else:
            logger.info(f"Epoch {self.current_epoch} finished — lr: {lr:.2e}")

    def validation_step(self, batch, batch_idx):
        x = batch['encoder_input']
        
        # Get ground truth strings (kern or lmx format)
        if 'kern_strings' in batch:
            gt_strings = batch['kern_strings']
            format_type = 'kern'
        elif 'lmx_strings' in batch:
            gt_strings = batch['lmx_strings']
            format_type = 'lmx'
        else:
            raise ValueError("Batch must contain either 'kern_strings' or 'lmx_strings'")

        self._format_type = format_type

        # Autoregressive prediction (one sample at a time since val batch_size=1)
        prediction, _ = self.model.predict(input=x, convert_to_str=False)

        # Debug: log first 3 samples to see what model actually generates
        if batch_idx < 3:
            logger.info(f"Val sample {batch_idx} ({format_type}): predicted {len(prediction)} tokens, "
                        f"first 20: {prediction[:20]}")
            logger.info(f"Val sample {batch_idx}: GT first 80 chars: {gt_strings[0][:80]}")

        # Convert token list to string based on format
        if format_type == 'kern':
            # Kern format uses special tokens
            # NOTE 2026-07-24: " ".join (not "".join) + space-delimited replaces, mirroring the
            # GT transform below. "".join collapsed within-cell chord/divisi notes (adjacent note
            # tokens like "2AA- 4GG-") into one blob "2AA-4GG-", mis-scoring even a PERFECT
            # prediction (~29% corpus-wide SER floor). Pred must be transformed identically to GT.
            pred_string = " ".join(prediction)
            pred_string = pred_string.replace(" <t> ", "\t").replace(" <b> ", "\n").replace(" <s> ", " ")
            
            # Ground truth is already a kern token string; convert to plain kern
            gt = gt_strings[0]
            gt = gt.replace(" <t> ", "\t").replace(" <b> ", "\n").replace(" <s> ", " ")
        else:
            # LMX format uses space-separated tokens
            pred_string = " ".join(prediction)
            gt = gt_strings[0]

        self.val_predictions.append(pred_string)
        self.val_ground_truths.append(gt)

    def on_validation_epoch_end(self):
        if not self.val_predictions:
            return

        cer, ser, ler = compute_poliphony_metrics(
            self.val_predictions, self.val_ground_truths,
            format_type=getattr(self, '_format_type', 'kern')
        )

        self.log('val_CER', cer, prog_bar=True)
        self.log('val_SER', ser, prog_bar=True)
        self.log('val_LER', ler, prog_bar=False)

        logger.info(f"Validation — SER: {ser:.4f}, CER: {cer:.4f}, LER: {ler:.4f} "
                    f"({len(self.val_predictions)} samples)")

        self.val_predictions.clear()
        self.val_ground_truths.clear()

    # ---- Test (same logic as validation, separate metrics) ----

    def on_test_start(self):
        self.test_predictions = []
        self.test_ground_truths = []
        self.test_musicxml_contents = []

    def test_step(self, batch, batch_idx):
        x = batch['encoder_input']

        # Get ground truth strings (kern or lmx format)
        if 'kern_strings' in batch:
            gt_strings = batch['kern_strings']
            format_type = 'kern'
        elif 'lmx_strings' in batch:
            gt_strings = batch['lmx_strings']
            format_type = 'lmx'
        else:
            raise ValueError("Batch must contain either 'kern_strings' or 'lmx_strings'")

        self._format_type = format_type

        prediction, _ = self.model.predict(input=x, convert_to_str=False)

        # Convert token list to string based on format
        if format_type == 'kern':
            # Kern format uses special tokens
            # NOTE 2026-07-24: " ".join (not "".join) + space-delimited replaces, mirroring the
            # GT transform below. "".join collapsed within-cell chord/divisi notes (adjacent note
            # tokens like "2AA- 4GG-") into one blob "2AA-4GG-", mis-scoring even a PERFECT
            # prediction (~29% corpus-wide SER floor). Pred must be transformed identically to GT.
            pred_string = " ".join(prediction)
            pred_string = pred_string.replace(" <t> ", "\t").replace(" <b> ", "\n").replace(" <s> ", " ")

            # Ground truth is already a kern token string; convert to plain kern
            gt = gt_strings[0]
            gt = gt.replace(" <t> ", "\t").replace(" <b> ", "\n").replace(" <s> ", " ")
        else:
            # LMX format uses space-separated tokens
            pred_string = " ".join(prediction)
            gt = gt_strings[0]

        self.test_predictions.append(pred_string)
        self.test_ground_truths.append(gt)

        # TEDn is a format-independent metric (MusicXML tree edit distance), computed on
        # top of SER/CER/LER so Kern and LMX runs become comparable (see
        # TOKENIZATION_IMPACT_ON_TRAINING_2026-07-01.md, Finding 4). Test-only: the report
        # measured ~8-23s/sample, too slow to run every validation epoch.
        if self.compute_tedn and 'musicxml_paths' in batch:
            with open(batch['musicxml_paths'][0], 'r', encoding='utf-8') as f:
                self.test_musicxml_contents.append(f.read())

    def on_test_epoch_end(self):
        if not self.test_predictions:
            return

        cer, ser, ler = compute_poliphony_metrics(
            self.test_predictions, self.test_ground_truths,
            format_type=getattr(self, '_format_type', 'kern')
        )

        self.log('test_CER', cer)
        self.log('test_SER', ser)
        self.log('test_LER', ler)

        log_msg = f"TEST — SER: {ser:.4f}, CER: {cer:.4f}, LER: {ler:.4f}"

        if self.compute_tedn and self.test_musicxml_contents:
            tedn = compute_tedn_metrics(
                self.test_predictions, self.test_musicxml_contents,
                format_type=getattr(self, '_format_type', 'kern'),
            )
            self.log('test_TEDn', tedn)
            log_msg += f", TEDn: {tedn:.4f}"

        logger.info(f"{log_msg} ({len(self.test_predictions)} samples)")

        self.test_predictions.clear()
        self.test_ground_truths.clear()
        self.test_musicxml_contents.clear()
