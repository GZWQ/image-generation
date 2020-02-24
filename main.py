import argparse
from model import PoseGAN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose guided image generation usign deformable skip layers')


    ##########Training Setting############
    parser.add_argument('--batch_size',default=10,type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument("--number_of_epochs", default=500, type=int, help="Number of training epochs")
    parser.add_argument('--dataset_name', default='cad60', choices=['market1501', 'cad60'])
    parser.add_argument("--display_ratio", default=1, type=int, help='Number of epochs between ploting')
    parser.add_argument("--checkpoint_ratio", default=3, type=int, help="Number of epochs between consecutive checkpoints")
    parser.add_argument("--pose_estimator", default='pose_estimator.h5',help='Pretrained model for cao pose estimator')
    parser.add_argument("--use_warp", default='none', choices=['none', 'full', 'mask', 'stn'],help="Type of warping skip layers to use.")
    parser.add_argument("--warp_agg", default='max', choices=['max', 'avg'],help="Type of aggregation.")
    parser.add_argument('--im_size',default=(128,64,3))
    parser.add_argument('--data_path',default='../../../../mydata/differentialPoseGan/data/') #/Users/daniel/Documents/JupiterGit/mydata/differentialPoseGan/data


    ##########Loss Setting###############
    parser.add_argument('--l1_penalty_weight',default=0,type=float)
    parser.add_argument('--gan_penalty_weight', default=1, type=float)
    parser.add_argument('--tv_penalty_weight', default=0, type=float, help='Weight of total variation loss')
    parser.add_argument('--lstruct_penalty_weight', default=0, type=float, help="Weight of lstruct")
    parser.add_argument("--mae_weight", default=0, type=int, help="Use nearest neighbour loss")

    parser.add_argument("--content_loss_layer", default='none', help='Name of content layer (vgg19) e.g. block4_conv1 or none')
    parser.add_argument("--nn_loss_area_size", default=1, type=int, help="Use nearest neighbour loss")

    ##########Directory Setting##########
    parser.add_argument('--output_dir',default='displayed_samples/')
    parser.add_argument("--checkpoints_dir", default="checkpoints/", help="Folder with checkpoints")

    args = parser.parse_args()
    model = PoseGAN(args)
    model.train()
