import matplotlib.pyplot as plt
import json
import numpy as np


def draw_classification_result(top1_path,top5_path=None,title='result',figuresize=(16,9)):
    '''
    Args:
        top1_path: str, json file
        top5_path: str, json file, same length with top1.json
        title: default 'result'
        figuresize: default (16,9)
    '''
    top1 = []
    with open(top1_path, 'r') as f:
        top1 = json.load(f)
    epoch = list(range(len(top1)))
    max_idx = np.argmax(top1)
    max_x = epoch[max_idx]
    max_y = top1[max_idx]
    
    plt.figure(figsize=figuresize)
    plt.title(title)
    plt.plot(epoch, top1, label='top1')
    plt.scatter(max_x, max_y, color='red', s=38)
    plt.annotate(f'best:{max_y:.3f}', xy=(max_x, max_y), xytext=(max_x-0.1, max_y-0.1),
            arrowprops=dict(facecolor='red', shrink=0.05))    
    
    if top5_path:
        top5= []
        with open(top5_path, 'r') as f:
            top5 = json.load(f)
        plt.plot(epoch, top5, label='top5')
        plt.legend()

    plt.savefig(f'{title}.jpg')


# draw_classification_result(top1_path='/Users/joriri/densenet_result_val02/top1.json', 
#                            top5_path='/Users/joriri/densenet_result_val02/top5.json')



