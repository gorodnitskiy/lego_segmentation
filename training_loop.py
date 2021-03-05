import torch
import pytorch_lightning as pl


def loss_custom(pred, target):
    return torch.nn.functional.cross_entropy(
        pred[(target != 255).unsqueeze(1).expand(pred.shape)],
        target[target != 255])


class LEGONet(pl.LightningModule):
    def __init__(
        self, model, loss, metric, optim: str = 'Adam',
        optim_lr: float = 1.0e-3, optim_weight_decay: float = 0.01,
        optim_momentum: float = 0.9, sched_mode: str = 'min',
        sched_factor: float = 0.2, sched_min_lr: float = 1e-6,
        sched_patience: int = 10, sched_verbose: bool = True
    ) -> None:
        super().__init__()
        self.model = model
        self._loss = loss
        self._metric = metric

        self._optim_nm = optim
        self._optim_lr = optim_lr
        self._optim_weight_decay = optim_weight_decay
        self._optim_momentum = optim_momentum

        self._sched_mode = sched_mode
        self._sched_min_lr = sched_min_lr
        self._sched_factor = sched_factor
        self._sched_patience = sched_patience
        self._sched_verbose = sched_verbose

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch['image'])
        loss_train = self._loss(pred, batch['mask'])
        metric_train = self._metric(pred, batch['mask'])

        self.log('IoU/train', metric_train, on_epoch=True)
        self.log('loss/train', loss_train, on_epoch=True)

        return loss_train

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch['image'])
        loss_val = self._loss(pred, batch['mask'])
        metric_val = self._metric(pred, batch['mask'])

        self.log('loss/valid', loss_val, on_epoch=True)
        self.log('IoU/valid', metric_val, on_epoch=True)

        return loss_val

    def configure_optimizers(self):
        if self._optim_nm == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self._optim_lr,
                weight_decay=self._optim_weight_decay)

        elif self._optim_nm == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self._optim_lr,
                weight_decay=self._optim_weight_decay)

        elif self._optim_nm == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self._optim_lr,
                momentum=self._optim_momentum,
                weight_decay=self._optim_weight_decay)
        else:
            raise ValueError('optimizers: Adam, AdamW or SGD optimizers')

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=self._sched_mode, min_lr=self._sched_min_lr,
            factor=self._sched_factor, patience=self._sched_patience,
            verbose=self._sched_verbose)

        return self.optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        epoch = args[0]
        batch_idx = args[1]

        val_accuracy = self.trainer.logged_metrics['IoU/valid']
        if epoch != 0 and batch_idx == 0:
            self.scheduler.step(val_accuracy)


class IoU(pl.metrics.Metric):
    def __init__(self, n_classes=21):
        super().__init__()
        self.n_classes = n_classes
        self.add_state("inter", default=torch.zeros([21]), dist_reduce_fx='sum')
        self.add_state("union", default=torch.zeros([21]), dist_reduce_fx='sum')

    def update(self, preds, target):
        res = preds.argmax(dim=1)
        for index in range(self.n_classes):
            truth = (target.cpu() == index)
            preds = (res == index)

            inter = truth.logical_and(preds.cpu())
            union = truth.logical_or(preds.cpu())

            self.inter[index] += inter.float().sum()
            self.union[index] += union.float().sum()

    def compute(self):
        return self.inter.sum() / self.union.sum()
