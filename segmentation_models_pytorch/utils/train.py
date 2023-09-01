import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y, w=None, loss_mode=0):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, loss_mode=0, sample_weight=None):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y, id in iterator:
                x, y = x.to(self.device), y.to(self.device)
                
                x_w = None
                
                if sample_weight is not None:
                    x_w = [sample_weight[k] for k in id]
                    # print ('\n', id, x_w, '\n')
                
                loss, y_pred = self.batch_update(x, y, x_w, loss_mode)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, w=None, loss_mode=0):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        
        # loss = self.loss(prediction, y)
        loss0 = self.loss(prediction, y[:, 0:1, :, :], w)          ### origin label
        
        loss1 = None                                                
        if loss_mode == 1 or loss_mode == 2 or loss_mode == 4:
            loss1 = self.loss(prediction, y[:, 1:2, :, :], w)       ### prior knowledge based soft label
    
        loss2 = None
        if loss_mode >= 3: 
            loss2 = self.loss(prediction, y[:, 2:3, :, :], w)       ### model distillation based soft label
        
        loss = None
        
        if loss_mode == 0:
            loss = loss0
        elif loss_mode == 1:
            loss = loss1
        elif loss_mode == 2:
            loss = (loss0 + loss1) / 2
        elif loss_mode == 3:
            loss = (loss0 + loss2) / 2
        elif loss_mode == 4:
            loss = 0.3 * loss0 + 0.4 * loss1 + 0.3 * loss2
        
        loss.backward()
        self.optimizer.step()
        return loss, prediction


# class ValidEpoch(Epoch):
#     def __init__(self, model, loss, metrics, device="cpu", verbose=True):
#         super().__init__(
#             model=model,
#             loss=loss,
#             metrics=metrics,
#             stage_name="valid",
#             device=device,
#             verbose=verbose,
#         )

#     def on_epoch_start(self):
#         self.model.eval()

#     def batch_update(self, x, y):
#         with torch.no_grad():
#             prediction = self.model.forward(x)
#             loss = self.loss(prediction, y)
#         return loss, prediction
