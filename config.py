from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=80, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('-b', '--batch-size', dest='batch_size', default=5, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=100, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=500, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')

parser.add_option('--ni', '--num_iterations', dest='num_iterations', default=3, type='int',
                  help='number of routing iterations (default: 3)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=10, type='int',
                  help='number of classes (default: 10)')

# For margin loss
parser.add_option('--mp', '--m_plus', dest='m_plus', default=0.9, type='float',
                  help='m+ parameter (default: 0.9)')
parser.add_option('--mm', '--m_minus', dest='m_minus', default=0.1, type='float',
                  help='m- parameter (default: 0.1)')
parser.add_option('--la', '--lambda_val', dest='lambda_val', default=0.5, type='float',
                  help='Down-weighting parameter for the absent class (default: 0.5)')
options, _ = parser.parse_args()
print()

