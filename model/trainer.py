import json
import os
import time
import numpy as np
import torch
from tqdm import tqdm

from lib.logger import (
    get_logger, 
    PD_Stats, 
)
from lib.utils import (
    get_log_dir, 
    get_model_params, 
    dwa,  
)
from lib.metrics import test_metrics

class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, args, prompt=None):
        super(Trainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        print('train', len(dataloader['train']))
        print('val', len(dataloader['val']))
        print('test', len(dataloader['test']))
        if prompt:
            self.train_loader = dataloader['val']
            self.val_loader = dataloader['test']
            self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.args = args
        self.prompt = prompt

        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)
        
        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.best_p_path = os.path.join(self.args.log_dir, 'best_p_model.pth')

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'), 
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

        with open(os.path.join(self.args.log_dir, 'config.json'), 'wt') as f:
            json.dump(args.__dict__, f)
    
    def train_epoch(self, epoch, loss_weights):
        # self.model.train()
        self.__train()

        total_loss = 0
        total_sep_loss = np.zeros(3) 
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            self.optimizer.zero_grad()
            
            # input shape: n,l,v,c; graph shape: v,v;
            repr1, repr2 = self.model(data, self.graph, self.prompt) # nvc
            loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)
            assert not torch.isnan(loss)
            loss.backward()

            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]), 
                    self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += sep_loss

        train_epoch_loss = total_loss/self.train_per_epoch
        total_sep_loss = total_sep_loss/self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss
    
    def val_epoch(self, epoch, val_dataloader, loss_weights):
        # self.model.eval()
        self.__eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                # print('data', data.shape)
                repr1, repr2 = self.model(data, self.graph, self.prompt)
                # print('repr1', repr1.shape)
                # print('repr2', repr2.shape)
                loss, sep_loss = self.model.loss(repr1, repr2, target, self.scaler, loss_weights)

                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        return val_loss

    def __train(self):
        self.model.train()
        if self.prompt:
            self.prompt.train()

    def __eval(self):
        self.model.eval()
        if self.prompt:
            self.prompt.eval()

    def train(self):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3) #(1.0, 1.0, 1.0)
        for epoch in range(1, self.args.epochs + 1):
            # dwa mechanism to balance optimization speed for different tasks
            if self.args.use_dwa:
                loss_tm2 = loss_tm1
                loss_tm1 = loss_t
                if (epoch == 1) or (epoch == 2):
                    loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                else:
                    loss_weights  = dwa(loss_tm1, loss_tm2, self.args.temp)
            # self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t = self.train_epoch(epoch, loss_weights)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)       
            if not self.args.debug:
                self.training_stats.update((epoch, train_epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch, 
                    "model": self.model.state_dict(), 
                    "optimizer": self.optimizer.state_dict(),
                }
                if self.prompt:
                    save_p_dict = self.prompt.state_dict()
                if not self.args.debug:
                    if self.prompt:
                        self.logger.info('**************Current best P model saved to {}'.format(self.best_p_path))
                        torch.save(save_p_dict, self.best_p_path)
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1

            # early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.args.early_stop_patience))
                break   

        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60), 
                        best_loss, 
                        best_epoch))
        
        # test
        if self.prompt:
            p_state_dict = save_p_dict if self.args.debug else torch.load(
                self.best_p_path, map_location=torch.device(self.args.device))
            self.prompt.load_state_dict(p_state_dict)
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler, 
                                self.graph, self.logger, self.args, self.prompt)
        results = {
            'best_val_loss': best_loss, 
            'best_val_epoch': best_epoch, 
            'test_results': test_results,
        }
        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args, prompt=None):
        model.eval()
        if prompt:
            prompt.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                repr1, repr2 = model(data, graph, prompt)
                pred_output = model.predict(repr1, repr2)

                y_true.append(target)
                y_pred.append(pred_output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        test_results = []
        # inflow
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape])
        # outflow 
        mae, mape = test_metrics(y_pred[..., 1], y_true[..., 1])
        logger.info("OUTFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape*100))
        test_results.append([mae, mape]) 

        return np.stack(test_results, axis=0)



        

