from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=100, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=100, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=1000, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=0, type='int',
                  help='number of data loading workers (default: 16)')
# For data
parser.add_option('--ih', '--img_h', dest='img_h', default=28, type='int',
                  help='input image height (default: 28)')
parser.add_option('--iw', '--img_w', dest='img_w', default=28, type='int',
                  help='input image width (default: 28)')
parser.add_option('--ic', '--img_c', dest='img_c', default=1, type='int',
                  help='number of input channels (default: 1)')

parser.add_option('--ni', '--num_iterations', dest='num_iterations', default=3, type='int',
                  help='number of routing iterations (default: 3)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=10, type='int',
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


# For decoder
parser.add_option('--ad', '--add_decoder', dest='add_decoder', default=False,
                  help='whether to use decoder or not')
parser.add_option('--h1', '--h1', dest='h1', default=512, type='int',
                  help='number of classes (default: 10)')
parser.add_option('--h2', '--h2', dest='h2', default=1024, type='int',
                  help='number of classes (default: 10)')


options, _ = parser.parse_args()
print()

