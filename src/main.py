import os
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
import network
import learning_approach
from dataset import get_loaders


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='Base Framework')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default='test', type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')

    # dataset args
    parser.add_argument('--data-type', default='combined', type=str, required=False,
                        help='Real or toy data (default=%(default)s)')
    parser.add_argument('--real-data-amount', default=40, type=int, required=False,
                        help='Real or toy data (default=%(default)s)')
    parser.add_argument('--retrain-toy', default=False, type=bool, required=False,
                        help='Should the toy data model be retrained (default=%(default)s)')
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=5, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')

    # training args
    parser.add_argument('--nepochs', default=30, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.01, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=3, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--momentum', default=0.95, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')

    # Args -- Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, momentum=args.momentum, wd=args.weight_decay)

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'

    # Path where the model is saved
    model_path = os.path.join(args.results_path, args.exp_name, 'trained_model.pth')

    # Network
    utils.seed_everything(seed=args.seed)
    net = network.simplenet(data_type=args.data_type)

    # Check if the model was previously saved and load it
    if args.retrain_toy is False and os.path.exists(model_path):
        print("Loading saved model from {}".format(model_path))
        net.load_state_dict(torch.load(model_path))
    else:
        print("No saved model found, starting training from scratch")

    # Learning Approach
    appr_args, extra_args = learning_approach.Learning_Appr.extra_parser(extra_args)
    utils.seed_everything(seed=args.seed)
    appr_kwargs = {**base_kwargs, **dict(**appr_args.__dict__)}
    appr = learning_approach.Learning_Appr(net, device, args.data_type, **appr_kwargs)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader = get_loaders(batch_sz=5,
                                                     num_work=args.num_workers,
                                                     pin_mem=args.pin_memory,
                                                     data_type=args.data_type,
                                                     real_data_amount=args.real_data_amount)

    # Get refining loaders
    trn_loader_2, val_loader_2, tst_loader_2 = get_loaders(batch_sz=args.batch_size,
                                                     num_work=args.num_workers,
                                                     pin_mem=args.pin_memory,
                                                     data_type="real",
                                                     real_data_amount=args.real_data_amount)


    # If starting from scratch, train the model
    if args.retrain_toy is True or not os.path.exists(model_path):
        print('*' * 108)
        print('Start training')
        print('*' * 108)
        net.to(device)
        appr.train(trn_loader, val_loader)
        # Save the model after training
        save_path = os.path.join(args.results_path, args.exp_name, 'trained_model.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(net.state_dict(), model_path)
        print('Model saved to {}'.format(model_path))
        print('-' * 108)

    # # Train
    # print('*' * 108)
    # print('Start training')
    # print('*' * 108)
    # net.to(device)
    # appr.train(trn_loader, val_loader)
    # save_path = os.path.join(args.results_path, args.exp_name, 'trained_model.pth')
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # torch.save(net.state_dict(), save_path)
    # print('Model saved to {}'.format(save_path))
    # print('-' * 108)

    # Refine
    model_path = os.path.join(args.results_path, args.exp_name, 'trained_model.pth')
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    print('*' * 108)
    print('Start refining')
    print('*' * 108)

    net.to(device)
    appr.train(trn_loader_2, val_loader_2)
    print('-' * 108)

    # Test
    test_loss, test_acc, predictions_list = appr.eval(tst_loader)
    print('>>> Test: loss={:.3f} | acc={:5.1f}% <<<'.format(test_loss, 100 * test_acc))

    # Save
    print('Save at ' + os.path.join(args.results_path, args.exp_name))
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    indices = np.array(range(3000))

    plt.figure(1)
    for i, sample in enumerate(tst_loader.dataset):
        signal = np.transpose(sample[0])

        if args.data_type == 'real':
            true_p = np.where(np.transpose(sample[1]) == 1)[0]
            true_r = np.where(np.transpose(sample[1]) == 2)[0]
            true_t = np.where(np.transpose(sample[1]) == 3)[0]

            plt.subplot(len(tst_loader.dataset), 1, i+1)
            plt.plot(indices, signal)
            plt.scatter(indices[true_p], signal[true_p], marker='D', s=20, color='red', label='P-wave')
            plt.scatter(indices[true_r], signal[true_r], marker='D', s=20, color='green', label='R-wave')
            plt.scatter(indices[true_t], signal[true_t], marker='D', s=20, color='pink', label='T-wave')

        elif args.data_type == 'toy' or 'combined':
            true_p = np.where(np.transpose(sample[1]) == 1)[0]
            true_q = np.where(np.transpose(sample[1]) == 2)[0]
            true_r = np.where(np.transpose(sample[1]) == 3)[0]
            true_s = np.where(np.transpose(sample[1]) == 4)[0]
            true_t = np.where(np.transpose(sample[1]) == 5)[0]


            plt.subplot(len(tst_loader.dataset), 1, i + 1)
            plt.plot(indices, signal)
            plt.scatter(indices[true_p], signal[true_p], marker='D', s=20, color='red', label='P-wave')
            plt.scatter(indices[true_q], signal[true_q], marker='D', s=20, color='blue', label='Q-wave')
            plt.scatter(indices[true_r], signal[true_r], marker='D', s=20, color='green', label='R-wave')
            plt.scatter(indices[true_s], signal[true_s], marker='D', s=20, color='orange', label='S-wave')
            plt.scatter(indices[true_t], signal[true_t], marker='D', s=20, color='pink', label='T-wave')



    plt.suptitle('Test signals with marked true peaks')
    plt.legend()
    plt.show()

    plt.figure(2)
    for i, sample in enumerate(tst_loader.dataset):
        signal = np.transpose(sample[0])

        if args.data_type == 'real':

            predicted_p = np.where(np.transpose(predictions_list[0][i] == 1))[0]
            predicted_r = np.where(np.transpose(predictions_list[0][i] == 2))[0]
            predicted_t = np.where(np.transpose(predictions_list[0][i] == 3))[0]

            plt.subplot(len(tst_loader.dataset), 1, i + 1)
            plt.plot(indices, signal)
            plt.scatter(indices[predicted_p], signal[predicted_p], marker='D', s=20, color='red', label='P-wave')
            plt.scatter(indices[predicted_r], signal[predicted_r], marker='D', s=20, color='green', label='R-wave')
            plt.scatter(indices[predicted_t], signal[predicted_t], marker='D', s=20, color='pink', label='T-wave')

        elif args.data_type == 'toy' or 'combined':
            predicted_p = np.where(np.transpose(predictions_list[0][i] == 1))[0]
            predicted_q = np.where(np.transpose(predictions_list[0][i] == 2))[0]
            predicted_r = np.where(np.transpose(predictions_list[0][i] == 3))[0]
            predicted_s = np.where(np.transpose(predictions_list[0][i] == 4))[0]
            predicted_t = np.where(np.transpose(predictions_list[0][i] == 5))[0]

            plt.subplot(len(tst_loader.dataset), 1, i + 1)
            plt.plot(indices, signal)
            plt.scatter(indices[predicted_p], signal[predicted_p], marker='D', s=20, color='red', label='P-wave')
            plt.scatter(indices[predicted_q], signal[predicted_q], marker='D', s=20, color='blue', label='Q-wave')
            plt.scatter(indices[predicted_r], signal[predicted_r], marker='D', s=20, color='green', label='R-wave')
            plt.scatter(indices[predicted_s], signal[predicted_s], marker='D', s=20, color='orange', label='S-wave')
            plt.scatter(indices[predicted_t], signal[predicted_t], marker='D', s=20, color='pink', label='T-wave')


    plt.suptitle('Test signals with marked predicted peaks')
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(range(len(appr.trn_loss_list)), appr.trn_loss_list, label='Train loss')
    plt.plot(range(len(appr.val_loss_list)), appr.val_loss_list, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return test_loss, test_acc


if __name__ == '__main__':
    main()
