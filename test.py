import os
import sys
import cv2
import time
import torch
import argparse

from utils.y2rgb import y2rgb
from utils import utils_image as util
from torch.utils.data import DataLoader
from data.dataset_wogt import TestDataset
from models.network_swinfusion import SwinFusion as net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='./logs/mixed/models/')
    parser.add_argument('--root_path', type=str,
                        default='./logs/mixed/')
    parser.add_argument('--iter_number', type=str,
                        default='20000')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--in_channel', type=int, default=1, help='3 means color image and 1 means gray image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model_path = os.path.join(args.model_path, args.iter_number + '_G.pth')
    if os.path.exists(model_path):
        print(f'loading model from {args.model_path}')
    else:
        print('Target model path: {} not existing!!!'.format(model_path))
        sys.exit()
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    save_dir = 'results_' + args.exp_name
    window_size = 8

    dataset_opts = {
        'LLVIP': {'task_path': '/data/TC-MoA-test/LLVIP_Test', 'subdir1': 'visible/test', 'subdir2': 'infrared/test',},
        'MEFB': {'task_path': '/data/TC-MoA-test/MEF_test', 'subdir1': 'input_B', 'subdir2': 'input_A'},
        'MFFB': {'task_path': '/data/TC-MoA-test/MFF_test', 'subdir1': 'input_B', 'subdir2': 'input_A'},
    }
    test_loaders = []

    for k, v in dataset_opts.items():
        test_set = TestDataset(v['task_path'], v['subdir1'], v['subdir2'])
        test_loaders.append(DataLoader(test_set, batch_size=1,
                                       shuffle=False, num_workers=1,
                                       drop_last=False, pin_memory=True))
    sum = 0
    for j, test_loader in enumerate(test_loaders):
        dataset_name = list(dataset_opts.keys())[j]
        os.makedirs(os.path.join(save_dir, dataset_name), exist_ok=True)
        for i, test_data in enumerate(test_loader):
            img_a = test_data['A'].to(device)
            img_b = test_data['B'].to(device)

            # inference
            start = time.time()
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_a.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img_a = torch.cat([img_a, torch.flip(img_a, [2])], 2)[:, :, :h_old + h_pad, :]
                img_a = torch.cat([img_a, torch.flip(img_a, [3])], 3)[:, :, :, :w_old + w_pad]
                img_b = torch.cat([img_b, torch.flip(img_b, [2])], 2)[:, :, :h_old + h_pad, :]
                img_b = torch.cat([img_b, torch.flip(img_b, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img_a, img_b, model, args, window_size)
                output = output[..., :h_old, :w_old]
                output = output.detach()[0].float().cpu()

            end = time.time()
            during = end - start
            sum += during
            print('during: ', during)
            output = util.tensor2uint(output)
            output = y2rgb(util.tensor2uint(test_data['A_rgb'])[:, :, ::-1],
                           util.tensor2uint(test_data['B_rgb'])[:, :, ::-1], output)
            save_name = os.path.join(save_dir, dataset_name, test_data['fname'][0])
            cv2.imwrite(save_name, output)
            print(
                "[{}/{}]  Saving fused image to : {}".format(i + 1, len(test_loader), save_name))

        print('runtime: ', sum/len(test_loader), ' s.')


def define_model(args):
    model = net(upscale=1, in_chans=args.in_channel, img_size=128, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler=None, resi_connection='oaf')

    param_key_g = 'params'
    model_path = os.path.join(args.model_path, args.iter_number + '_E.pth')
    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                          strict=True)
    total_parameters = sum([param.nelement() for param in model.parameters()])
    print("Total parameters: %.2fM" % (total_parameters / 1e6))
    return model


def test(img_a, img_b, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_a, img_b)
    else:
        # test the image tile by tile
        b, c, h, w = img_a.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_a)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_a[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()
