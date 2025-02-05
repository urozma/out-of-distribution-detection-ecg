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
from add_qs import add_qs_peaks
from scipy.stats import wilcoxon
from sklearn import metrics

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

    # scenario args
    parser.add_argument('--transfer', default=True, type=bool, required=False,
                        help='Scenario (default=%(default)s)')
    parser.add_argument('--no-transfer-type', default='real', type=str, required=False,
                        help='Which data should be used in the no transfer case: real or toy (default=%(default)s)')
    parser.add_argument('--real-data-amount', default=136, type=int, required=False,
                        help='Real or toy data (default=%(default)s)')
    parser.add_argument('--add-qs', default=True, type=bool, required=False,
                        help='Add QS peaks before training (default=%(default)s)')
    parser.add_argument('--add-qs-after', default=False, type=bool, required=False,
                        help='Add QS peaks to test results for plotting (default=%(default)s)')

    # retrain args
    parser.add_argument('--retrain', default=False, type=bool, required=False,
                        help='Should the model be retrained from scratch(default=%(default)s)')
    parser.add_argument('--retrain-finetune', default=False, type=bool, required=False,
                        help='Should the model be finetuned again (default=%(default)s)')

    # dataset args
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=5, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')

    # training args
    parser.add_argument('--nepochs', default=40, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.01, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--momentum', default=0.95, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=1e-3, type=float, required=False,
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
    finetuned_model_path = os.path.join(args.results_path, args.exp_name, 'finetuned_model.pth')

    # Define data type for training according to scenario
    if args.transfer is True:
        data_type = 'toy'
    else:
        data_type = args.no_transfer_type

    # Network
    utils.seed_everything(seed=args.seed)
    net = network.simplenet(data_type=data_type)

    # Check if the model was previously saved and load it
    if args.retrain is False and os.path.exists(model_path):
        print('Loading saved trained model from {}'.format(model_path))
        net.load_state_dict(torch.load(model_path))
    else:
        print('No saved model found, starting training from scratch')

    # Learning Approach
    appr_args, extra_args = learning_approach.Learning_Appr.extra_parser(extra_args)
    utils.seed_everything(seed=args.seed)
    appr_kwargs = {**base_kwargs, **dict(**appr_args.__dict__)}
    appr = learning_approach.Learning_Appr(net, device, data_type, **appr_kwargs)
    if len(np.sort(list(vars(appr_args).keys()))) > 0:
        print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    assert len(extra_args) == 0, 'Unused args: {}'.format(' '.join(extra_args))
    print('-' * 108)

    # Get training loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader = get_loaders(batch_sz=args.batch_size,
                                                     num_work=args.num_workers,
                                                     pin_mem=args.pin_memory,
                                                     data_type=data_type,
                                                     real_data_amount=args.real_data_amount,
                                                     add_qs=args.add_qs)

    # Get finetuning loaders
    trn_loader_ft, val_loader_ft, tst_loader_ft = get_loaders(batch_sz=args.batch_size,
                                                     num_work=args.num_workers,
                                                     pin_mem=args.pin_memory,
                                                     data_type='real',
                                                     real_data_amount=args.real_data_amount,
                                                     add_qs=args.add_qs)

    # Get unhealthy data loaders
    _,_,ood_tst_loader = get_loaders(batch_sz=args.batch_size,
                                                     num_work=args.num_workers,
                                                     pin_mem=args.pin_memory,
                                                     data_type='real',
                                                     real_data_amount=args.real_data_amount,
                                                     health=['Sinus tachycardia', 'Sinus bradycardia', 'Sinus arrhythmia','Atrial fibrillation'],
                                                     add_qs=args.add_qs)


    # If starting from scratch, train the model
    if args.retrain is True or not os.path.exists(model_path):
        print('*' * 108)
        print('Start training')
        print('*' * 108)
        net.to(device)
        appr.train(trn_loader, val_loader)
        # Save the model after training
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(net.state_dict(), model_path)
        print('Model saved to {}'.format(model_path))
        print('-' * 108)

    # Finetune
    if args.transfer is True:
        if args.retrain_finetune is False and os.path.exists(finetuned_model_path):
            print('Loading saved finetuned model from {}'.format(finetuned_model_path))
            net.load_state_dict(torch.load(finetuned_model_path))
        else:
            print('No saved model found, starting refining')

        if args.retrain_finetune is True or not os.path.exists(model_path):
            model_path = os.path.join(args.results_path, args.exp_name, 'finetuned_model.pth')
            net.load_state_dict(torch.load(model_path))

            net.to(device)
            print('*' * 108)
            print('Start finetuning')
            print('*' * 108)

            net.to(device)
            appr.train(trn_loader_ft, val_loader_ft)
            # Save the model after refining
            os.makedirs(os.path.dirname(finetuned_model_path), exist_ok=True)
            torch.save(net.state_dict(), finetuned_model_path)
            print('Model saved to {}'.format(model_path))
            print('-' * 108)
    print('-' * 108)

    # Test
    test_loss, test_acc, predictions_list, latent_healthy = appr.eval(tst_loader_ft)
    uh_test_loss, uh_test_acc, uh_predictions_list, latent_unhealthy = appr.eval(ood_tst_loader)
    _, _, _, latent_trn = appr.eval(trn_loader_ft)

    if args.add_qs_after is True:
        predictions_list = add_qs_peaks(tst_loader_ft.dataset, predictions_list)

    print('>>> Test (H): loss={:.3f} | acc={:5.1f}% <<<'.format(test_loss, 100 * test_acc))
    print('>>> Test (D): loss={:.3f} | acc={:5.1f}% <<<'.format(uh_test_loss, 100 * uh_test_acc))

    # Save
    print('Save at ' + os.path.join(args.results_path, args.exp_name))
    print('[Elapsed time ={:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')
    print('-' * 108)

    # Flatten the latent space arrays if they are multi-dimensional
    latent_healthy_flat = np.concatenate([x.flatten() for x in latent_healthy])
    latent_unhealthy_flat = np.concatenate([x.flatten() for x in latent_unhealthy])
    latent_trn_flat = np.concatenate([x.flatten() for x in latent_trn])

    def flatten_and_concatenate(latents_list):
        # Flatten each tensor and concatenate into a single tensor
        flattened = [tensor.view(tensor.size(0), -1) for tensor in latents_list]
        concatenated = torch.cat(flattened, dim=0)
        return concatenated

    def compute_distances(train_latents, test_latents):
        distances = torch.cdist(test_latents, train_latents, p=2)  # p=2 for euclidian
        return distances

    def rank_samples(distances):
        min_distances = distances.min(dim=1)[0]
        # sorted_indices = torch.argsort(min_distances)
        return min_distances #sorted_indices, min_distances

    def calculate_metrics(y_score, y_true):
        # calculate curves
        fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
        prec, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        # pick position closer to TPR at 95%
        idx = (np.abs(tpr - 0.95)).argmin()
        # calculate Detection Error
        det_error = 0.5 * (1 - tpr[idx]) + 0.5 * fpr[idx]

        auroc = metrics.roc_auc_score(y_true, y_score)
        aupr_in = metrics.auc(recall, prec)

        # print the metrics
        print('---')
        print('TPR {}%, FPR {}% '.format(np.round(tpr[idx] * 100, decimals=2), np.round(fpr[idx] * 100, decimals=2)))
        print('Detection Error {} '.format(det_error * 100, decimals=2))
        print('AUROC {}%'.format(np.round(auroc * 100, decimals=2)))
        print('AUPR In {}%'.format(np.round(aupr_in * 100, decimals=2)))
        # interchange positive and negative classes to print AUPR-out
        y_true = 1 - y_true
        y_score = 1.0 - y_score
        prec, recall, _ = metrics.precision_recall_curve(y_true, y_score)
        aupr_out = metrics.auc(recall, prec)
        print('AUPR Out {}%'.format(np.round(aupr_out * 100, decimals=2)))

        return tpr[idx], fpr[idx], det_error, auroc, aupr_in, aupr_out


    # Compute distances
    train_latents = flatten_and_concatenate(latent_trn)
    test_latents = flatten_and_concatenate(latent_healthy+latent_unhealthy)
    dist_matrix = compute_distances(train_latents, test_latents)

    # Rank
    min_distances = rank_samples(dist_matrix)

    # Create labels for the test samples: 0 for healthy, 1 for unhealthy
    healthy_labels = torch.zeros(int(len(latent_healthy_flat)/3000/8))
    unhealthy_labels = torch.ones(int(len(latent_unhealthy_flat)/3000/8))
    true_labels = torch.cat([healthy_labels, unhealthy_labels])

    # Evaluate
    tpr, fpr, det_error, auroc, aupr_in, aupr_out = calculate_metrics(min_distances, true_labels)

    if len(latent_healthy_flat) > len(latent_unhealthy_flat):
        latent_healthy_flat = latent_healthy_flat[:len(latent_unhealthy_flat)]
    elif len(latent_unhealthy_flat) > len(latent_healthy_flat):
        latent_unhealthy_flat = latent_unhealthy_flat[:len(latent_healthy_flat)]

    # Perform the Mann-Whitney U Test
    stat, p_value = wilcoxon(x=latent_healthy_flat, y=latent_unhealthy_flat, alternative='two-sided')

    print('Wilcoxon Test')
    print('Number of healthy ECGs compared:', int(len(latent_healthy_flat)/3000/8))
    print('Number of unhealthy ECGs compared:', int(len(latent_unhealthy_flat)/3000/8))
    print('P-value:', p_value)

    # Interpret the p-value
    alpha = 0.05  # significance level
    if p_value > alpha:
        print('No significant difference between the distributions')
    else:
        print('Significant difference between the distributions')
    print('=' * 108)

    indices = np.array(range(3000))/500

    plt.figure(num=1, figsize=(8, 8))
    for i in range(5):
        sample = tst_loader_ft.dataset[i]
        signal = np.transpose(sample[0])

        true_p = np.where(np.transpose(sample[1]) == 1)[0]
        true_q = np.where(np.transpose(sample[1]) == 2)[0]
        true_r = np.where(np.transpose(sample[1]) == 3)[0]
        true_s = np.where(np.transpose(sample[1]) == 4)[0]
        true_t = np.where(np.transpose(sample[1]) == 5)[0]


        plt.subplot(5, 1, i + 1)
        plt.plot(indices, signal)
        plt.scatter(indices[true_p], signal[true_p], marker='D', s=20, color='green', label='P-wave')
        plt.scatter(indices[true_q], signal[true_q], marker='D', s=20, color='orange', label='Q-wave')
        plt.scatter(indices[true_r], signal[true_r], marker='D', s=20, color='red', label='R-wave')
        plt.scatter(indices[true_s], signal[true_s], marker='D', s=20, color='blue', label='S-wave')
        plt.scatter(indices[true_t], signal[true_t], marker='D', s=20, color='fuchsia', label='T-wave')

        plt.grid(True, which='major', axis='both', linestyle='-', color='gray')
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', linestyle='--', color='lightgray')


        plt.xticks(np.arange(0, 10, 1))
        plt.ylim(-8, 8)
        plt.xlim(0, 6)

    plt.suptitle('Ground Truth', y=0.93)
    plt.xlabel('Time in seconds')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(2, figsize=(8, 8))

    for i in range(5):
        sample = tst_loader_ft.dataset[i]
        signal = np.transpose(sample[0])

        predicted_p = np.where(np.transpose(predictions_list[0][i] == 1))[0]
        predicted_q = np.where(np.transpose(predictions_list[0][i] == 2))[0]
        predicted_r = np.where(np.transpose(predictions_list[0][i] == 3))[0]
        predicted_s = np.where(np.transpose(predictions_list[0][i] == 4))[0]
        predicted_t = np.where(np.transpose(predictions_list[0][i] == 5))[0]

        plt.subplot(5, 1, i + 1)
        plt.plot(indices, signal)
        plt.scatter(indices[predicted_p], signal[predicted_p], marker='D', s=20, color='green', label='P-wave')
        plt.scatter(indices[predicted_q], signal[predicted_q], marker='D', s=20, color='orange', label='Q-wave')
        plt.scatter(indices[predicted_r], signal[predicted_r], marker='D', s=20, color='red', label='R-wave')
        plt.scatter(indices[predicted_s], signal[predicted_s], marker='D', s=20, color='blue', label='S-wave')
        plt.scatter(indices[predicted_t], signal[predicted_t], marker='D', s=20, color='fuchsia', label='T-wave')

        plt.grid(True, which='major', axis='both', linestyle='-', color='gray')
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', linestyle='--', color='lightgray')

        # Set the x-axis major tick marks to appear every 1 ms
        plt.xticks(np.arange(0, 10, 1))
        plt.ylim(-8, 8)
        plt.xlim(0, 6)

    plt.suptitle('Predictions', y=0.93)
    plt.xlabel('Time in seconds')
    plt.legend(loc="lower right")
    plt.show()


    plt.figure(num=3, figsize=(8, 8))
    for i in range(5):
        sample = ood_tst_loader.dataset[i]
        signal = np.transpose(sample[0]) * 1

        true_p = np.where(np.transpose(sample[1]) == 1)[0]
        true_q = np.where(np.transpose(sample[1]) == 2)[0]
        true_r = np.where(np.transpose(sample[1]) == 3)[0]
        true_s = np.where(np.transpose(sample[1]) == 4)[0]
        true_t = np.where(np.transpose(sample[1]) == 5)[0]

        plt.subplot(5, 1, i + 1)
        plt.plot(indices, signal)
        plt.scatter(indices[true_p], signal[true_p], marker='D', s=20, color='green', label='P-wave')
        plt.scatter(indices[true_q], signal[true_q], marker='D', s=20, color='orange', label='Q-wave')
        plt.scatter(indices[true_r], signal[true_r], marker='D', s=20, color='red', label='R-wave')
        plt.scatter(indices[true_s], signal[true_s], marker='D', s=20, color='blue', label='S-wave')
        plt.scatter(indices[true_t], signal[true_t], marker='D', s=20, color='fuchsia', label='T-wave')

        plt.grid(True, which='major', axis='both', linestyle='-', color='gray')
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', linestyle='--', color='lightgray')

        # Set the x-axis major tick marks to appear every 1 ms
        plt.xticks(np.arange(0, 10, 1))
        plt.ylim(-8, 8)
        plt.xlim(0, 6)

    plt.suptitle('Unhealhy ECGs', y=0.93)
    plt.xlabel('Time in seconds')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(num=4, figsize=(8, 8))
    for i in range(5):
        sample = ood_tst_loader.dataset[i]
        signal = np.transpose(sample[0])

        predicted_p = np.where(np.transpose(uh_predictions_list[0][i] == 1))[0]
        predicted_q = np.where(np.transpose(uh_predictions_list[0][i] == 2))[0]
        predicted_r = np.where(np.transpose(uh_predictions_list[0][i] == 3))[0]
        predicted_s = np.where(np.transpose(uh_predictions_list[0][i] == 4))[0]
        predicted_t = np.where(np.transpose(uh_predictions_list[0][i] == 5))[0]

        plt.subplot(5, 1, i + 1)
        plt.plot(indices, signal)
        plt.scatter(indices[predicted_p], signal[predicted_p], marker='D', s=20, color='green', label='P-wave')
        plt.scatter(indices[predicted_q], signal[predicted_q], marker='D', s=20, color='orange', label='Q-wave')
        plt.scatter(indices[predicted_r], signal[predicted_r], marker='D', s=20, color='red', label='R-wave')
        plt.scatter(indices[predicted_s], signal[predicted_s], marker='D', s=20, color='blue', label='S-wave')
        plt.scatter(indices[predicted_t], signal[predicted_t], marker='D', s=20, color='fuchsia', label='T-wave')

        plt.grid(True, which='major', axis='both', linestyle='-', color='gray')
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', linestyle='--', color='lightgray')

        # Set the x-axis major tick marks to appear every 1 ms

        plt.xticks(np.arange(0, 10, 1))
        plt.ylim(-8, 8)
        plt.xlim(0, 6)

    plt.suptitle('Unhealhy ECGs Predictions', y=0.93)
    plt.xlabel('Time in seconds')
    plt.legend(loc="lower right")
    plt.show()

    return test_loss, test_acc, uh_test_loss, uh_test_acc, p_value, tpr, fpr, det_error, auroc, aupr_in, aupr_out

if __name__ == '__main__':
    num_iterations = 10  # Number of times to repeat the training and evaluation
    all_test_accs, all_uh_test_accs = [], []  # List to store test accuracy of each iteration
    all_test_losses, all_uh_test_losses = [], []  # List to store test loss of each iteration
    all_p_values = []
    all_tpr, all_fpr, all_det_error, all_auroc, all_aupr_in, all_aupr_out = [], [], [], [], [], []

    for iteration in range(num_iterations):
        print(f'Starting iteration {iteration+1}/{num_iterations}')
        test_loss, test_acc, uh_test_loss, uh_test_acc,  p_value,  tpr, fpr, det_error, auroc, aupr_in, aupr_out = main()  # Assuming main returns test_loss and test_acc
        all_test_accs.append(test_acc)
        all_test_losses.append(test_loss)
        all_uh_test_accs.append(uh_test_acc)
        all_uh_test_losses.append(uh_test_loss)
        all_p_values.append(p_value)
        all_tpr.append(tpr)
        all_fpr.append(fpr)
        all_det_error.append(det_error)
        all_auroc.append(auroc)
        all_aupr_in.append(aupr_in)
        all_aupr_out.append(aupr_out)

    # After all iterations are done, save the results
    results_path = '../results'  # Or any path you prefer
    np.save(os.path.join(results_path, 'all_test_accs.npy'), np.array(all_test_accs))
    np.save(os.path.join(results_path, 'all_test_losses.npy'), np.array(all_test_losses))
    np.save(os.path.join(results_path, 'all_uh_test_accs.npy'), np.array(all_uh_test_accs))
    np.save(os.path.join(results_path, 'all_uh_test_losses.npy'), np.array(all_uh_test_losses))
    np.save(os.path.join(results_path, 'all_p_values.npy'), np.array(all_p_values))
    np.save(os.path.join(results_path, 'all_tpr.npy'), np.array(all_tpr))
    np.save(os.path.join(results_path, 'all_fpr.npy'), np.array(all_fpr))
    np.save(os.path.join(results_path, 'all_det_error.npy'), np.array(all_det_error))
    np.save(os.path.join(results_path, 'all_auroc.npy'), np.array(all_auroc))
    np.save(os.path.join(results_path, 'all_aupr_in.npy'), np.array(all_aupr_in))
    np.save(os.path.join(results_path, 'all_aupr_out.npy'), np.array(all_aupr_out))


    print('All iterations completed and results saved.')

    # Load the saved results
    results_path = '../results'  # Adjust this if you used a different path
    all_test_accs = np.load(os.path.join(results_path, 'all_test_accs.npy'))
    all_test_losses = np.load(os.path.join(results_path, 'all_test_losses.npy'))
    all_uh_test_accs = np.load(os.path.join(results_path, 'all_uh_test_accs.npy'))
    all_uh_test_losses = np.load(os.path.join(results_path, 'all_uh_test_losses.npy'))
    all_p_values = np.load(os.path.join(results_path, 'all_p_values.npy'))
    all_tpr = np.load(os.path.join(results_path, 'all_tpr.npy'))
    all_fpr = np.load(os.path.join(results_path, 'all_fpr.npy'))
    all_det_error = np.load(os.path.join(results_path, 'all_det_error.npy'))
    all_auroc = np.load(os.path.join(results_path, 'all_auroc.npy'))
    all_aupr_in = np.load(os.path.join(results_path, 'all_aupr_in.npy'))
    all_aupr_out = np.load(os.path.join(results_path, 'all_aupr_out.npy'))


    # Calculate averages
    average_accuracy = np.mean(all_test_accs)
    average_loss = np.mean(all_test_losses)
    average_uh_accuracy = np.mean(all_uh_test_accs)
    average_uh_loss = np.mean(all_uh_test_losses)
    average_p_value = np.mean(all_p_values)
    average_tpr = np.mean(all_tpr)
    average_fpr = np.mean(all_fpr)
    average_det_error = np.mean(all_det_error)
    average_auroc = np.mean(all_auroc)
    average_aupr_in = np.mean(all_aupr_in)
    average_aupr_out = np.mean(all_aupr_out)


    # Print averages
    print('Average Test Accuracy (H):{:5.1f}%'.format(100 * average_accuracy))
    print('Average Test Accuracy (D):{:5.1f}%'.format(100 * average_uh_accuracy))
    print('Average Test Loss (H): {:.3f}'.format(average_loss))
    print('Average Test Loss (D): {:.3f}'.format(average_uh_loss))
    print('Average P-value: ',average_p_value)
    print('Average TPR score: {}%'.format(np.round(average_tpr * 100, decimals=2)))
    print('Average FPR score: {}%'.format(np.round(average_fpr * 100, decimals=2)))
    print('Average Detection Error: {}%'.format(np.round(average_det_error * 100, decimals=2)))
    print('Average AUROC score: {}%'.format(np.round(average_auroc * 100, decimals=2)))
    print('Average AUPR In score: {}%'.format(np.round(average_aupr_in * 100, decimals=2)))
    print('Average AUPR Out score: {}%'.format(np.round(average_aupr_out * 100, decimals=2)))