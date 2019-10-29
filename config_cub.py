from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=2, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=10, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=100, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=0, type='int',
                  help='number of data loading workers (default: 16)')
# For data
parser.add_option('--dn', '--data_name', dest='data_name', default='cub',
                  help='mnist, fashion_mnist, t_mnist, c_mnist, cub (default: mnist)')

parser.add_option('--ih', '--img_h', dest='img_h', default=448, type='int',
                  help='input image height (default: 28)')
parser.add_option('--iw', '--img_w', dest='img_w', default=448, type='int',
                  help='input image width (default: 28)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 1)')

parser.add_option('--ni', '--num_iterations', dest='num_iterations', default=3, type='int',
                  help='number of routing iterations (default: 3)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=200, type='int',
                  help='number of classes (default: 10)')

# For loss
parser.add_option('--mp', '--m_plus', dest='m_plus', default=0.9, type='float',
                  help='m+ parameter (default: 0.9)')
parser.add_option('--mm', '--m_minus', dest='m_minus', default=0.1, type='float',
                  help='m- parameter (default: 0.1)')
parser.add_option('--la', '--lambda_val', dest='lambda_val', default=0.5, type='float',
                  help='Down-weighting parameter for the absent class (default: 0.5)')
parser.add_option('--al', '--alpha', dest='alpha', default=0.0005, type='float',
                  help='Regularization coefficient to scale down the reconstruction loss (default: 0.0005)')

parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

# For CapsNet
parser.add_option('--fe', '--feature_extractor', dest='feature_extractor', default='resnet',
                  help='densenet, resnet(default: resnet)')

parser.add_option('--f1', '--f1', dest='f1', default=512, type='int',
                  help='number of filters for the conv1 layer (default: 256)')
parser.add_option('--k1', '--k1', dest='k1', default=9, type='int',
                  help='filter size of the conv1 layer (default: 9)')

parser.add_option('--f2', '--f2', dest='f2', default=256, type='int',
                  help='number of filters for the primary capsule layer (default: 256)')
parser.add_option('--k2', '--k2', dest='k2', default=9, type='int',
                  help='filter size of the primary capsule layer (default: 9)')

parser.add_option('--pcd', '--primary_cap_dim', dest='primary_cap_dim', default=8, type='int',
                  help='dimension of each primary capsule (default: 8)')
parser.add_option('--dcd', '--digit_cap_dim', dest='digit_cap_dim', default=16, type='int',
                  help='dimension of each digit capsule (default: 16)')

# For decoder
parser.add_option('--ad', '--add_decoder', dest='add_decoder', default=False,
                  help='whether to use decoder or not')
parser.add_option('--h1', '--h1', dest='h1', default=512, type='int',
                  help='number of hidden units of the first hidden layer (default: 512)')
parser.add_option('--h2', '--h2', dest='h2', default=1024, type='int',
                  help='number of hidden units of the first hidden layer (default: 1024)')

# For Options
parser.add_option('--ws', '--weight_share', dest='share_weight', default=True,
                  help='whether to share W among child capsules of the same type (default: True)')

parser.add_option('--ca', '--add_coord', dest='add_coord', default=False,
                  help='whether to add coordinates to the primary capsules output or not (default: False)')
parser.add_option('--sc', '--shuffle_coords', dest='shuffle_coords', default=False,
                  help='whether to shuffle the coordinates or not (default: False)')

parser.add_option('--usn', '--use_simnet', dest='use_simnet', default=False,
                  help='whether to use the Similarity Network or not (default: False)')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/capsule_network_pytorch/save/20191023_181535/models/25200.ckpt',
                  help='path to load a .ckpt model')


options, _ = parser.parse_args()
