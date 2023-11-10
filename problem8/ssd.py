# https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/
import torch
import os

ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

ssd_model.to('cuda')
ssd_model.eval()

from matplotlib import pyplot as plt
import matplotlib.patches as patches

directory = 'grocerystore'
filenames = os.listdir(directory)
output_directory = 'ssd'

inputs = [utils.prepare_input(os.path.join(directory, jpg)) for jpg in filenames]
tensor = utils.prepare_tensor(inputs)

with torch.no_grad():
    detections_batch = ssd_model(tensor)

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

classes_to_labels = utils.get_coco_object_dictionary()

for image_idx, filename in enumerate(filenames):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
    output_path = os.path.join(output_directory, filename)
    plt.savefig(output_path)
    plt.close(fig)