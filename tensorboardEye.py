import matplotlib.pyplot as plt
import numpy as np
import os
def images_to_probs(net, train_x_):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(train_x_)
    # convert output probabilities to predicted class
    preds = output.cpu().detach().numpy()
    #print('hi')
    return preds


def plot_classes_preds(net, train_x_, train_y_):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds = images_to_probs(net, train_x_)
    gt = train_y_.cpu().detach().numpy()
    # plot the images in the batch, along with predicted and true labels

    fig =  plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('Predicted')
    plt.plot(np.squeeze(preds[0]))
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(np.squeeze(gt[0]))
    ax2.set_title('Ground truth')
    #plt.show()
    return fig

def imshow_on_tensorboard(image_, gt_, sr_):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''

    image_cpu = image_.cpu().detach().numpy()
    gt = gt_.cpu().detach().numpy()
    sr = sr_.cpu().detach().numpy()
    # plot the images in the batch, along with predicted and true labels

    fig =  plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('Input')
    plt.imshow(np.squeeze(image_cpu[0]))
    #
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('Ground truth')
    plt.imshow(np.squeeze(gt[0]))
    #
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title('Prediction')
    plt.imshow(np.squeeze(sr[0]))

    #plt.show()
    return fig

def save_on_local(image_, gt_, sr_, locs_, _batch):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''

    image_cpu = image_.cpu().detach().numpy()
    gt = gt_.cpu().detach().numpy()
    sr = sr_.cpu().detach().numpy()
    # plot the images in the batch, along with predicted and true labels

    fig =  plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('Input')
    plt.imshow(np.squeeze(image_cpu[0]))
    #
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('Ground truth')
    plt.imshow(np.squeeze(gt[0]))
    #
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title('Prediction')
    plt.imshow(np.squeeze(sr[0]))

    #plt.show()
    import os
    if not os.path.exists(locs_):
        os.makedirs(locs_)

    fileName = os.path.join(locs_, 'ValidationSet_' + str(_batch) + '.png')
    plt.savefig(fileName)
    plt.close()
    return