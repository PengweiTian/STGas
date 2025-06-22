import copy
import json
import os
import warnings

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from ..model.arch import build_model

from STGas.util import gather_results, mkdir, show_batch_img
from STGas.optim import build_optimizer


# 构建训练任务
class TrainingTask(LightningModule):

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "STGas"

        self.weight_averager = None

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch["img"] = [item.to(self.device) for item in batch["img"]]
        predict = self.forward(batch["img"])
        results = self.model.head.post_process(predict, batch)
        return results

    def training_step(self, batch, batch_idx):
        batch["img"] = [item.to(self.device) for item in batch["img"]]
        predict, loss, loss_states = self.model.forward_train(batch)

        # log train losses
        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                batch_idx + self.cfg.log.interval,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            log_msg += f"loss_total:{loss:.4f}"
            self.logger.info(log_msg)
        # for name, param in self.model.named_parameters():
        #     if "temporal" in name:
        #         if param.grad is not None:
        #             print(f"{name}: grad norm = {param.grad.norm()}")
        #         else:
        #             print(f"{name}: grad is None")  # 该参数未参与梯度计算
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))
        self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        batch["img"] = [item.to(self.device) for item in batch["img"]]
        predict, loss, loss_states = self.model.forward_train(batch)

        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                batch_idx + self.cfg.log.interval,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        dets = self.model.head.post_process(predict, batch)
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs
        """
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "STGas_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            json.dump(res_json, open(json_path, "w"))

            if self.cfg.test_mode == "val":
                eval_results = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            self.logger.info("Skip test on rank {}".format(self.local_rank))

    def configure_optimizers(self):
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(self.model, optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        return optimizer

    # def optimizer_step(
    #         self,
    #         epoch=None,
    #         batch_idx=None,
    #         optimizer=None,
    #         optimizer_idx=None,
    #         optimizer_closure=None,
    #         on_tpu=None,
    #         using_native_amp=None,
    #         using_lbfgs=None,
    # ):
    #     """
    #     Performs a single optimization step (parameter update).
    #     Args:
    #         epoch: Current epoch
    #         batch_idx: Index of current batch
    #         optimizer: A PyTorch optimizer
    #         optimizer_idx: If you used multiple optimizers this indexes into that list.
    #         optimizer_closure: closure for all optimizers
    #         on_tpu: true if TPU backward is required
    #         using_native_amp: True if using native amp
    #         using_lbfgs: True if the matching optimizer is lbfgs
    #     """
    #     # warm up lr
    #     if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
    #         if self.cfg.schedule.warmup.name == "constant":
    #             warmup_lr = (
    #                     self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
    #             )
    #         elif self.cfg.schedule.warmup.name == "linear":
    #             k = (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (
    #                     1 - self.cfg.schedule.warmup.ratio
    #             )
    #             warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
    #         elif self.cfg.schedule.warmup.name == "exp":
    #             k = self.cfg.schedule.warmup.ratio ** (
    #                     1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
    #             )
    #             warmup_lr = self.cfg.schedule.optimizer.lr * k
    #         else:
    #             raise Exception("Unsupported warm up type!")
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = warmup_lr
    #
    #     # for name, param in self.model.named_parameters():
    #     #     if "CTDFF" in name:
    #     #         if param.grad is not None:
    #     #             print(f"{name}: grad norm = {param.grad.norm()}")
    #     #         else:
    #     #             print(f"{name}: grad is None")  # 该参数未参与梯度计算
    #
    #     # update params
    #     # optimizer.step(closure=optimizer_closure)
    #
    #
    #
    #     loss = optimizer_closure()
    #     # print("Loss:", loss)
    #     # print("Loss requires grad?", loss.requires_grad)
    #     # print("Loss grad_fn:", loss.grad_fn)
    #
    #     optimizer.step()
    #     optimizer.zero_grad()


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def scalar_summary(self, tag, phase, value, step):
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)

    # ------------Hooks-----------------
    def on_train_epoch_start(self):
        self.model.set_epoch(self.current_epoch)
