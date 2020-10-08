import argparse


class Options:
    """
    Options and settings for training, debugging and evaluation of FHDR model.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # training options
        self.parser.add_argument(
            "--batch_size", type=int, default=2, help="batch size for training network."
        )
        self.parser.add_argument(
            "--epochs", type=int, default=200, help="number of epochs"
        )

        self.parser.add_argument(
            "--lr", type=float, default=0.0002, help="learning rate"
        )
        self.parser.add_argument(
            "--lr_decay_after",
            type=int,
            default=100,
            help="linear decay of learning rate starts at this epoch",
        )

        self.parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training: load the latest model",
        )
        self.parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )

        self.parser.add_argument(
            "--iter",
            type=int,
            default="1",
            help="number of iterations for feedback mechanisms (refer to paper)",
        )

        # debugging options
        self.parser.add_argument(
            "--print_model", action="store_true", help="print model"
        )
        self.parser.add_argument(
            "--save_ckpt_after",
            type=int,
            default=2,
            help="number of epochs after which checkpoints are saved",
        )
        self.parser.add_argument(
            "--log_after",
            type=int,
            default=500,
            help="number of batches after which batch, loss is logged",
        )
        self.parser.add_argument(
            "--save_results_after",
            type=int,
            default=1000,
            help="number of batches after which results are saved",
        )

        # testing options
        self.parser.add_argument(
            "--ckpt_path",
            type=str,
            default="./checkpoints/latest.ckpt",
            help="path of checkpoint to be loaded",
        )
        self.parser.add_argument(
            "--log_scores",
            action="store_true",
            help="log PSNR, SSIM scores at evaluation",
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
