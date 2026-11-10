import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os
import torch

# Importing from local modules
from tools import write2csv, setup_paths, setup_seed, log_metrics, Logger
from dataset import get_data
from method import PGCLIPTrainer

setup_seed(111)

def train(args):
    # Configurations
    epochs = args.epoch
    learning_rate = args.learning_rate
    learning_rate_textpromptlearner = args.learning_rate_textpromptlearner
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_fig = args.save_fig

    # Set up paths
    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = setup_paths(args)
    # Logger
    logger = Logger(log_path)

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    logger.info('Model name: {:}'.format(model_name))

    config_path = os.path.join('./model_configs', f'{args.model}.json')

    # Prepare model
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    # Set up the feature hierarchy
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = PGCLIPTrainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=learning_rate,
        learning_rate_textpromptlearner=learning_rate_textpromptlearner,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters
    ).to(device)

    train_data_cls_names, train_data, train_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=True)

    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.testing_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False)

    logger.info('Data Root: training, {:}; testing, {:}'.format(train_data_root, test_data_root))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Typically, we use MVTec or VisA as the validation set. The best model from this validation
    # process is then used for zero-shot anomaly detection on novel categories.
    best_f1 = -1e1

    for epoch in tqdm(range(epochs)):
        loss, classification_loss, seg_loss, loss_ab,loss_bias = model.train_epoch(train_dataloader)

        # Logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss))
            logger.info('percific loss : classification_loss:{:.4f}, seg_loss:{:.4f}, loss_ab:{:.4f}, loss_bias:{:.4f}'.format(classification_loss, seg_loss, loss_ab,loss_bias))
            tensorboard_logger.add_scalar('loss', loss, epoch)

        # Validation
        if (epoch + 1) % args.valid_freq == 0 or (epoch == epochs - 1):
            if epoch == epochs - 1:
                save_fig_flag = save_fig
            else:
                save_fig_flag = False

            logger.info('=============================Testing ====================================')
            metric_dict = model.evaluation(
                test_dataloader,
                test_data_cls_names,
                save_fig_flag,
                image_dir,
                ifdino=False,
            )

            log_metrics(
                metric_dict,
                logger,
                tensorboard_logger,
                epoch
            )

            f1_px = metric_dict['Average']['f1_px']

            # Save best
            if f1_px > best_f1:
                for k in metric_dict.keys():
                    write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

                ckp_path_best = ckp_path + '_best.pth'
                model.save(ckp_path_best)
                best_f1 = f1_px



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PG-CLIP", add_help=True)

    # Paths and configurations
    parser.add_argument("--training_data", type=str, default=["mvtec", "colondb"], nargs='+',
                        help="Datasets for training (default: ['mvtec', 'colondb'])")
    parser.add_argument("--testing_data", type=str, default="visa", help="Dataset for testing (default: 'visa')")

    parser.add_argument("--save_path", type=str, default='./workspaces',
                        help="Directory to save results (default: './workspaces')")

    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used (default: 'ViT-L-14-336')")

    parser.add_argument("--save_fig", type=str2bool, default=False,
                        help="Save figures for visualizations (default: False)")
    parser.add_argument("--ckt_path", type=str, default='', help="Path to the pre-trained model (default: '')")

    # Hyper-parameters
    parser.add_argument("--exp_indx", type=int, default=0, help="Index of the experiment (default: 0)")
    parser.add_argument("--epoch", type=int, default=8, help="Number of epochs (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")

    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images (default: 518)")
    parser.add_argument("--print_freq", type=int, default=1, help="Frequency of print statements (default: 1)")
    parser.add_argument("--valid_freq", type=int, default=1, help="Frequency of validation (default: 1)")

    # Prompting parameters
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting (default: 4)")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting (default: 5)")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'],
                        help="Type of prompting. 'S' for Static, 'D' for Dynamic, 'SD' for both (default: 'SD')")
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'],
                        help="Branch of prompting. 'V' for Visual, 'L' for Language, 'VL' for both (default: 'VL')")

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation. If False, original class embedding is used (default: True)")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters (default: 20)")

    # Add new parameters
    #parser.add_argument("--text_norm_weight", type=float, default=0.01, help="weight of orthogonal_loss (FN/FA) (default: 0.1)")
    parser.add_argument("--learning_rate_textpromptlearner", type=float, default=0.01, help="Textpromptlearner learning rate  (default: 0.001)")

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. Please set --batch_size to 1.")

    train(args)

