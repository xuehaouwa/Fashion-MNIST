import argparse
import torch
from torchvision import datasets, transforms
import time
from torch.autograd import Variable


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--save_folder', default='saved_model/v3_da_2/v3.pkl', type=str, help="path for saved trained model")
    parser.add_argument('-bs', '--batch_size', default=128, type=int, help="batch size")
    return parser.parse_args()


class Test:
    use_gpu = torch.cuda.is_available()

    def __init__(self, args):
        self.args = args
        self.net = None
        self.test_dataloader = None

    def load_model(self):


        if self.use_gpu:
            self.net = torch.load(self.args.save_folder)
            self.net.cuda()
        else:
            self.net = torch.load(self.args.save_folder, map_location="cpu")


    def build_dataloader(self):
        test_transform = transforms.Compose([transforms.ToTensor()])

        self.test_dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=test_transform),
            batch_size=self.args.batch_size, shuffle=False)

    def testing(self):

        self.net.eval()
        correct_num = 0
        for i, (batch_x, batch_y) in enumerate(self.test_dataloader):
            if self.use_gpu:
                data = Variable(batch_x).cuda()
            else:
                data = Variable(batch_x)
            target = batch_y
            output = self.net(data)
            if self.use_gpu:
                output = output.cpu()
            predicted = output.argmax(dim=1, keepdim=True)
            correct_num += predicted.eq(target.view_as(predicted)).sum().item()

        print(f"Total Correct Num: {correct_num} Accuracy: {correct_num / 10000}")


    def print_model_parm_nums(self):
        model = self.net
        total = sum([param.nelement() for param in model.parameters()])
        print(f' Number of params: {total / 1e6}M')


if __name__ == "__main__":
    args = get_args()

    tester = Test(args=args)
    tester.build_dataloader()
    tester.load_model()
    tester.print_model_parm_nums()
    testing_start = time.time()
    tester.testing()
    print(f"Testing Time: {time.time() - testing_start}")

