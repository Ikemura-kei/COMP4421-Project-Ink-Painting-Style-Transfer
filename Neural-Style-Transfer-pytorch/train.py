import torch
import torch.nn as nn

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import model
import utils
import kornia

import argparse

nst = model.nst_model
loader = utils.loader
imshow = utils.imshow
device = 'cuda:0'

def main(args):
    content_img = args.content_img
    style_img = args.style_img
    size = args.size
    steps = args.steps
    c_weight = args.c_weight
    s_weight = args.s_weight

    edge_weight = 10

    content_img, style_img = loader(content_img, style_img, size = size)
    content_img = content_img.to(device)
    style_img = style_img.to(device)
    input_img = content_img.clone().to(device) # just noise array is fine
    # input_img = input_img.requires_grad_()

    # test = input_img.clone().requires_grad_()
    # print("before test requires gradient:", test.requires_grad)
    # zero = torch.zeros(test.shape, requires_grad=True).to(device)
    # one = torch.ones(test.shape, requires_grad=True).to(device)
    # test = torch.where(test > 1, one, zero)
    # print("after test requires gradient:", test.requires_grad)

    # print("before:",input_img.requires_grad)
    # mag, edge = kornia.filters.canny(input_img, hysteresis=False)
    # print("after", edge.requires_grad)
    # imshow(edge, title="canny")
    
    model, style_losses, content_losses, edge_losses  = nst(content_img, style_img, input_img)

    optimizer = optim.LBFGS([input_img.requires_grad_()])

    step = [0]
    while step[0] <= steps:
        
        # method inside while
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            output = model(input_img)

            cl = 0
            sl = 0
            el = 0

            for c_loss in content_losses:
                cl += c_loss.loss * c_weight
            for s_loss in style_losses:
                sl += s_loss.loss * s_weight
            for e_loss in edge_losses:
                el += e_loss.loss * edge_weight

            

            loss = cl + sl +el
            loss.backward()

            if step[0] % 50 == 0:
                print('Step : {}'. format(step))
                print('Style Loss : {:3f} Content Loss: {:3f} Edge Loss: {:3f}'.format(
                    sl.item(), cl.item(), el.item()))

            step[0] += 1

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0,1)
    # print(input_img.shape)
    return input_img

    imshow(content_img, title = 'Input image')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img', type=str, default = 'images/395_A.png')
    parser.add_argument('--style_img', type=str, default = 'images/395_B.png')
    parser.add_argument('--size', type=int, default = 512, help='if you want to get more clear pictures, increase the size')
    parser.add_argument('--steps', type=int, default = 300 )
    parser.add_argument('--c_weight', type=int, default = 1, help='weighting factor for content reconstruction')
    parser.add_argument('--s_weight', type=int, default = 100000, help='weighting factor for style reconstruction')

    args = parser.parse_args()
    print(args)
    output = main(args)
    
    plt.figure()
    imshow(output, title = 'Output Image')
    plt.pause(5)
    plt.show()