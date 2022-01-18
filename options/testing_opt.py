
from options.base_opt import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def init_experiment_params(self, parser):
        parser = BaseOptions.init_experiment_params(self, parser)  # define shared options
        
        parser.add_argument('--pretrained_path', type=str, default='./check_points/', help='Path to pretrained model.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')

        # rewrite devalue values
        parser.set_defaults(model='test')
        self.is_train = False

        return parser