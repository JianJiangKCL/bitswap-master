from argparse import ArgumentParser


def custom_args(parser: ArgumentParser):
	parser.add_argument('--nz', default=2, type=int, help="number of latent variables (layers), greater or equal to 1")
	parser.add_argument('--zchannels', default=4, type=int, help="number of channels for the latent variables")
	parser.add_argument('--nprocessing', default=4, type=int, help='number of processing layers')
	parser.add_argument('--interval', default=100, type=int, help="interval for logging/printing of relevant values")

	parser.add_argument('--blocks', default=4, type=int, help="number of ResNet blocks")
	parser.add_argument('--width', default=64, type=int,
	                    help="number of channels in the convolutions in the ResNet blocks")
	parser.add_argument('--dropout', default=0.2, type=float, help="dropout rate of the hidden units")
	parser.add_argument('--kernel', default=3, type=int,
	                    help="size of the convolutional filter (kernel) in the ResNet blocks")

	parser.add_argument('--schedule', default=1, type=float, help="learning rate schedule: yes (1) or no (0)")
	parser.add_argument('--is_finetune', default=0, type=int)
	parser.add_argument('--vae_path', default=None, type=str)
	parser.add_argument('--code_level', default=None, type=str)
	parser.add_argument('--decay', default=0.9995, type=float,
	                    help="decay of the learning rate when using learning rate schedule")
	parser.add_argument('--tag', default='full', type=str)