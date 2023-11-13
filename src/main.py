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
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=16, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')

    # training args
    parser.add_argument('--nepochs', default=1, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.01, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float, required=False,
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

    # # Args -- CUDA
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # else:
    #     print('WARNING: [CUDA unavailable] Using CPU instead!')
    #     device = 'cpu'

    device = 'cpu'

    # Network
    utils.seed_everything(seed=args.seed)
    net = network.simplenet()

    # Learning Approach
    appr_args, extra_args = learning_approach.Learning_Appr.extra_parser(extra_args)
    utils.seed_everything(seed=args.seed)
    appr_kwargs = {**base_kwargs, **dict(**appr_args.__dict__)}
    appr = learning_approach.Learning_Appr(net, device, **appr_kwargs)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, reconstruction_info = get_loaders(batch_sz=args.batch_size,
                                                                                  num_work=args.num_workers,
                                                                                  pin_mem=args.pin_memory)

    # Task
    print('*' * 108)
    print('Start training')
    print('*' * 108)

    # Train
    net.to(device)
    appr.train(trn_loader, val_loader)
    print('-' * 108)

    # Test
    test_loss, test_acc, predictions_list = appr.eval(tst_loader)
    print('>>> Test: loss={:.3f} | acc={:5.1f}% <<<'.format(test_loss, 100 * test_acc))

    # Save
    print('Save at ' + os.path.join(args.results_path, args.exp_name))
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    def transform_for_reconstruction(memory_dataset):
        transformed_signals = []
        transformed_targets = []

        # Iterate over each sample in the dataset
        for i in range(len(memory_dataset)):
            signal_tensor, target_tensor = memory_dataset[i]

            # Reshape and convert tensors to numpy arrays if necessary
            signal_array = signal_tensor.squeeze().numpy()  # Removing unnecessary dimensions
            target_array = target_tensor.squeeze().numpy()

            transformed_signals.append(signal_array)
            transformed_targets.append(target_array)

        # Ensure the output is a tuple of two numpy arrays
        return (np.array(transformed_signals), np.array(transformed_targets))

    def reconstruct_signals(segmented_data, reconstruction_info):
        original_signals = []
        original_targets = []

        # Unpack the segmented signals and targets
        segmented_signals, segmented_targets = segmented_data

        # Iterate over segmented signals and targets with reconstruction info
        for i in range(len(segmented_signals)):
            segment = segmented_signals[i]
            target_segment = segmented_targets[i]
            start, end = reconstruction_info[i]

            original_length = end - start
            original_signal = segment[:original_length]
            original_target = target_segment[:original_length]

            # Assuming the signals are continuous, append the reconstructed segments
            if not original_signals or len(original_signals[-1]) != start:
                original_signals.append(original_signal)
                original_targets.append(original_target)
            else:
                original_signals[-1] = np.append(original_signals[-1], original_signal)
                original_targets[-1] = np.append(original_targets[-1], original_target)

        return (np.array(original_signals), np.array(original_targets))

    # t = np.linspace(0, 5, int(300 * 5), endpoint=False)
    # indices = np.array(range(len(np.transpose(tst_loader.dataset[0][0]))))
    indices = np.array(range(500))
    transformed = transform_for_reconstruction(tst_loader.dataset)
    print(len(transformed[0][0]))
    reconstructed_tuple = reconstruct_signals(transformed, reconstruction_info)


    plt.figure(1)
    for i, sample in enumerate(reconstructed_tuple):
        # signal = np.transpose(sample[0])
        #
        # true_p = np.where(np.transpose(sample[1]) == 1)[0]
        # #true_q = np.where(np.transpose(sample[1]) == 2)[0]
        # true_r = np.where(np.transpose(sample[1]) == 2)[0]
        # #true_s = np.where(np.transpose(sample[1]) == 4)[0]
        # true_t = np.where(np.transpose(sample[1]) == 3)[0]
        #
        # plt.subplot(len(tst_loader.dataset), 1, i+1)
        # plt.plot(indices, signal)
        # plt.scatter(indices[true_p], signal[true_p], marker='D', s=20, color='red', label='P-wave')
        # #plt.scatter(indices[true_q], signal[true_q], marker='D', s=20, color='orange', label='Q-wave')
        # plt.scatter(indices[true_r], signal[true_r], marker='D', s=20, color='green', label='R-wave')
        # #plt.scatter(indices[true_s], signal[true_s], marker='D', s=20, color='blue', label='S-wave')
        # plt.scatter(indices[true_t], signal[true_t], marker='D', s=20, color='pink', label='T-wave')



        signal = np.transpose(reconstructed_tuple[0])
        signal = reconstructed_tuple[0]


        # true_p = np.where(np.transpose(reconstructed_tuple[1]) == 1)[0]
        # # true_q = np.where(np.transpose(sample[1]) == 2)[0]
        # true_r = np.where(np.transpose(reconstructed_tuple[1]) == 2)[0]
        # # true_s = np.where(np.transpose(sample[1]) == 4)[0]
        # true_t = np.where(np.transpose(reconstructed_tuple[1]) == 3)[0]

        true_p = np.where((reconstructed_tuple[1]) == 1)[0]
        true_r = np.where((reconstructed_tuple[1]) == 2)[0]
        true_t = np.where((reconstructed_tuple[1]) == 3)[0]

        plt.subplot(len(tst_loader.dataset), 1, i + 1)
        plt.plot(indices, signal)
        plt.scatter(indices[true_p], signal[true_p], marker='D', s=20, color='red', label='P-wave')
        # plt.scatter(indices[true_q], signal[true_q], marker='D', s=20, color='orange', label='Q-wave')
        plt.scatter(indices[true_r], signal[true_r], marker='D', s=20, color='green', label='R-wave')
        # plt.scatter(indices[true_s], signal[true_s], marker='D', s=20, color='blue', label='S-wave')
        plt.scatter(indices[true_t], signal[true_t], marker='D', s=20, color='pink', label='T-wave')

    plt.suptitle('Test signals with marked true peaks')
    plt.legend()
    plt.show()

    plt.figure(2)
    for i, sample in enumerate(tst_loader.dataset):
        signal = np.transpose(sample[0])

        # true_peaks = np.where(np.transpose(sample[1]) == 3)[0]
        predicted_p = np.where(np.transpose(predictions_list[0][i] == 1))[0]
        # predicted_q = np.where(np.transpose(predictions_list[0][i] == 2))[0]
        predicted_r = np.where(np.transpose(predictions_list[0][i] == 2))[0]
        # predicted_s = np.where(np.transpose(predictions_list[0][i] == 4))[0]
        predicted_t = np.where(np.transpose(predictions_list[0][i] == 3))[0]

        plt.subplot(len(tst_loader.dataset), 1, i + 1)
        plt.plot(indices, signal)
        plt.scatter(indices[predicted_p], signal[predicted_p], marker='D', s=20, color='red', label='P-wave')
        # plt.scatter(indices[predicted_q], signal[predicted_q], marker='D', s=20, color='orange', label='Q-wave')
        plt.scatter(indices[predicted_r], signal[predicted_r], marker='D', s=20, color='green', label='R-wave')
        # plt.scatter(indices[predicted_s], signal[predicted_s], marker='D', s=20, color='blue', label='S-wave')
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
