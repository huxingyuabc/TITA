import os
import math
import time
import random
import logging
import argparse
import warnings
import numpy as np

from torch.utils.data import DataLoader
from data.dataset_wogt import MultiDataLoader, Mixed_Dataset, Mixed_Equal_Dataset

from models.model_plain import ModelPlain
from models.loss_vif import fusion_loss_vif
from models.loss_mef import fusion_loss_mef
from models.loss_mff import fusion_loss_mff
from models.loss_uni import fusion_loss_general

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from utils.famo_util import *
from utils.weight_methods import WeightMethods


warnings.filterwarnings("ignore")

def main():
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config.json', help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

    for key, path in opt['path'].items():
        print("Saving path: ", path)
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # automatically find latest checkpoint and resume
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    start_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)
    print('Start from step: ', start_step)
    current_step = start_step

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
        print('Random seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_type = dataset_opt['dataset_type'].lower()
            if dataset_type in ['mixed_equal']:
                train_loaders = []
                for k, v in dataset_opt['trainsets'].items():
                    train_set = Mixed_Equal_Dataset(k, v, dataset_opt['patch_size'])
                    train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
                    logger.info('Task: {}, Number of train images: {:,d}, iters: {:,d}'.format(k, len(train_set), train_size))

                    train_loader = DataLoader(train_set,
                                              batch_size=dataset_opt['dataloader_batch_size'],
                                              shuffle=dataset_opt['dataloader_shuffle'],
                                              num_workers=dataset_opt['dataloader_num_workers'],
                                              drop_last=True,
                                              pin_memory=True)
                    train_loaders.append(train_loader)
            elif dataset_type in ['mixed']:
                train_set = Mixed_Dataset(dataset_opt)

                train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
                train_loaders = [train_loader]
            else:
                raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

            multi_loader = MultiDataLoader(train_loaders)
            print('Dataset [{:s} - {:s}] is created.'.format(train_set.__class__.__name__, dataset_opt['name']))

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model_opt = opt['model']  # one input: L

    if model_opt == 'plain':
        model = ModelPlain(opt, device)
    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model_opt))

    print('Training model [{:s}] is created.'.format(model.__class__.__name__))

    model.init_train()

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    N = 3   # number of tasks

    ###################################### famo begin ######################################
    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(gamma=opt['train']['gamma'],
                                                                           method_params_lr=opt['train']['method_params_lr'])

    weight_method = WeightMethods(method='famo', n_tasks=N, device=device, **weight_methods_parameters['famo'])

    # loss
    G_lossfns = {}
    for k, G_lossfn_type in opt['train']['G_lossfn_types'].items():
        if G_lossfn_type == 'uni':
            G_lossfns[k] = fusion_loss_general().to(device)
        elif G_lossfn_type == 'mef':
            G_lossfns[k] = fusion_loss_mef().to(device)
        elif G_lossfn_type == 'vif':
            G_lossfns[k] = fusion_loss_vif().to(device)
        elif G_lossfn_type == 'mff':
            G_lossfns[k] = fusion_loss_mff().to(device)

    start_time = time.time()

    for epoch in range(opt['train']['epoch']):  # keep running
        if current_step >= opt['train']['iteration']:
            break

        for i in range(len(multi_loader)):
            batch = next(multi_loader)
            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.G_scheduler.step(current_step)
            model.G_optimizer.zero_grad()

            task_losses = []
            for t, train_data in enumerate(batch):
                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                model.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                model.netG_forward()

                G_lossfn_type = list(opt['train']['G_lossfn_types'].values())[t]
                G_lossfn = list(G_lossfns.values())[t]

                # calculate loss
                if G_lossfn_type in ['uni']:
                    weight_A, weight_B = G_lossfn.get_weights(model.A, model.B)
                    total_loss, loss_l1, loss_SSIM = G_lossfn(model.A, model.B, model.E, weight_A, weight_B)
                    task_losses.append(opt['train']['G_lossfn_weight'] * total_loss)
                elif G_lossfn_type in ['mef', 'mff', 'vif', 'nir', 'med']:
                    total_loss, loss_text, loss_int, loss_ssim = G_lossfn(model.A, model.B, model.E)
                    task_losses.append(opt['train']['G_lossfn_weight'] * total_loss)
                else:
                    task_losses.append(opt['train']['G_lossfn_weight'] * G_lossfn(model.E, model.GT))

            losses = torch.stack(task_losses)

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.netG.parameters()),
                task_specific_parameters=None,
                last_shared_parameters=None,
                representation=None,
            )

            model.G_optimizer.step()

            with torch.no_grad():
                task_losses = []
                for t, train_data in enumerate(batch):
                    # -------------------------------
                    # 2) feed patch pairs
                    # -------------------------------
                    model.feed_data(train_data)

                    # -------------------------------
                    # 3) optimize parameters
                    # -------------------------------
                    model.netG_forward()
                    G_lossfn_type = list(opt['train']['G_lossfn_types'].values())[t]
                    G_lossfn = list(G_lossfns.values())[t]

                    ## constructe loss function
                    if G_lossfn_type in ['uni']:
                        weight_A, weight_B = G_lossfn.get_weights(model.A, model.B)
                        total_loss, loss_l1, loss_SSIM = G_lossfn(model.A, model.B, model.E, weight_A, weight_B)
                        task_losses.append(opt['train']['G_lossfn_weight'] * total_loss)
                    elif G_lossfn_type in ['mef', 'mff', 'vif', 'nir', 'med']:
                        total_loss, loss_text, loss_int, loss_ssim = G_lossfn(model.A, model.B, model.E)
                        task_losses.append(opt['train']['G_lossfn_weight'] * total_loss)
                    else:
                        task_losses.append(opt['train']['G_lossfn_weight'] * G_lossfn(model.E, model.GT))

                new_losses = torch.stack(task_losses)
                weight_method.method.update(new_losses.detach())

                for t, G_loss in enumerate(task_losses):
                    model.log_dict['G_loss_'+list(G_lossfns.keys())[t]] = G_loss.item()
                    # report to TensorBoard
                    model.writer.add_scalar(
                        tag='G_loss/' + list(G_lossfns.keys())[t],
                        scalar_value=G_loss.item(),
                        global_step=current_step
                    )

                if opt['train']['E_decay'] > 0:
                    model.update_E(opt['train']['E_decay'])

                # -------------------------------
                # 4) training information
                # -------------------------------
                if current_step % opt['train']['checkpoint_print'] == 0:
                    logs = model.current_log()
                    eta = ((time.time()-start_time) / (current_step - start_step) * (opt['train']['iteration'] - current_step))
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, eta:{:2d}h{:2d}m> '.format(epoch, current_step,
                                                                              model.current_learning_rate(), int(eta//3600),
                                                                              int(eta%3600//60))
                    for k, v in logs.items():  # merge log information into message
                        message += '{:s}: {:.3e} '.format(k, v)
                    logger.info(message)

                # -------------------------------
                # 5) save model
                # -------------------------------
                if current_step % opt['train']['checkpoint_save'] == 0:
                    save_dir = opt['path']['models']
                    save_filename = '{}_{}.pth'.format(current_step, 'E')
                    save_path = os.path.join(save_dir, save_filename)
                    logger.info('Saving the model. Save path is:{}'.format(save_path))
                    model.save(current_step)

if __name__ == '__main__':
    main()
