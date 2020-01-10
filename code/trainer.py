import os
from decimal import Decimal

import utility
import errors

import torch
from torch.autograd import Variable

import matplotlib.pyplot as pl
import numpy as np
from skimage.io import imsave

def normalise01(a):
    """
    normalise a 2D array to [0, 1]
    """    
    b = (a - a.min()) / (a.max() - a.min())    
    return b

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        # loader:a dataset Class defined in main():loader=data.Data(args)
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.train_loader
        self.loader_test = loader.test_loader
        self.loader_results = loader.results_loader
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.': # load: file name to laod
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()
        self.error_last = 1e8

    def train(self):
        self.scheduler.step() # defines the learning rate decay
#        self.loss.step() # only executes with adverserial loss
        # for writing a log file:
        epoch = self.scheduler.last_epoch + 1 # get the index of last_epoch
        lr = self.scheduler.get_lr()[0] # get learning rate

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
            )
        self.loss.start_log() 
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # this lr is for lr images
        # why loader train for the original script has 4 arguments?
        running_loss = 0.0        
        for batch, sampled_batch in enumerate(self.loader_train):
            
            lr = sampled_batch['im_lr']
            hr = sampled_batch['im_hr']
            lr, hr = self.prepare([lr, hr]) # moves to cuda (potentially obsolete line)
 
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            # get input and scale and produce sr image
            sr, sr_ = self.model(lr) # self.model(lr, 2) forward path
#            sr = torch.clamp(sr, 0, 1)
            loss = self.loss(sr, sr_, hr)
            # skip the batch that has large error
            if 1:# loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
#            else:
#                print('Skip this batch {}! (Loss: {})'.format(
#                    batch + 1, loss.item()
#                ))
            running_loss += loss.item()
            timer_model.hold()
            # log training status every "print_every" batch (100 default)
            if (batch + 1) % self.args.print_every == 0:
                # replace "self.ckp.write_log" with print
                self.ckp.write_log(r'[batch_size:{}/dataset_size:{}]' 
                      r'    running_loss:'
                      r' {}    time_model+time_data:{:.1f}+{:.1f}s'
                      r''.format((batch + 1) * self.args.batch_size,
                                 len(self.loader_train.dataset),
                                 running_loss,
                                 timer_model.release(),
                                 timer_data.release()))
            timer_data.tic()
        self.loss.end_log(len(self.loader_train))
        # error_last is the total loss (I think)
        self.error_last = self.loss.log[-1, -1]
        
    def epoch(self):
        return self.scheduler.last_epoch + 1

    def test(self): # test or validation
        epoch = self.epoch()
        self.ckp.write_log('\nEvaluation:')
        scale = 2
        self.ckp.add_log(torch.zeros(1)) #(torch.zeros(1, len(self.scale)))
        self.model.eval()
        timer_test = utility.timer()
        with torch.no_grad():
                eval_acc = 0 #psnr loss
                valid_loss = 0 # total loss based on the training loss
                for im_idx, im_dict  in enumerate(self.loader_test, 1):
                    lr = im_dict['im_lr']
                    hr = im_dict['im_hr']
                    lr, hr = self.prepare([lr, hr])
                    sr, sr_ = self.model(lr)
                    sr = torch.clamp(sr, 0, 1)
                    sr_ = torch.clamp(sr_, 0, 1)
                    self.lr_valid = np.average(lr[0,:,:,:].permute(1, 2, 0).cpu().numpy(),
                                               axis=2)
                    self.hr_valid = hr[0,:,:,:].permute(1, 2, 0).cpu().numpy()
                    self.sr_valid = sr[0,:,:,:].permute(1, 2, 0).cpu().detach().numpy()
#                   sr = utility.quantize(sr, self.args.rgb_range) 
                    save_list = [sr]
                    # do some processing on sr, hr or modify find_psnr()
                    eval_acc += errors.find_psnr(sr, hr)
                    save_list.extend([lr, hr])
                    loss = self.loss.valid_loss(sr, sr_, hr)
                    valid_loss += loss.item()
                    # save the sr images of the last epoch
                    if self.args.save_results and epoch==self.args.epochs:
                        self.ckp.save_results("image_{}_sr".format(im_idx),
                                              save_list, scale)
                self.ckp.log_accuracy[-1] = (valid_loss / len(self.loader_test))
                self.ckp.log[-1] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1],
                        best[0].item(),
                        epoch
                    )
                )
        # ckp.save saves loss and model and plot_loss defined in the
        # Checkpoint class
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
#            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
            self.ckp.save(self, epoch, is_best=False)

    def test_model(self, test_model_dir): # test or validation
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1)) #(torch.zeros(1, len(self.scale)))
        self.model.eval()
        with torch.no_grad():
                eval_acc = 0
                for im_idx, im_dict  in enumerate(self.loader_results, 1):
                    lr = im_dict['im_lr']
                    hr = im_dict['im_hr']
                    lr, hr = self.prepare([lr, hr])
                    sr, sr_ = self.model(lr)
                    #sr = torch.clamp(sr, 0, 1)	
                    eval_acc += errors.find_psnr(sr, hr)
                    if True:                    
                        im_sr = np.float64(normalise01(sr[0,:,:,:].permute(1, 2, 0).cpu().numpy()))
                        im_sr = im_sr /im_sr.max()
                        im_sr = np.uint8(im_sr * 255)
                        imsave(test_model_dir + '/im_sr_{}.tiff'.format(im_idx + 275), im_sr)
                    print("Image: {}".format(im_idx))
                psnr = eval_acc / len(self.loader_test)
        return psnr

    # move tensors to GPU
    def prepare(self, l, volatile=False): 
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device, dtype=torch.float)
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
