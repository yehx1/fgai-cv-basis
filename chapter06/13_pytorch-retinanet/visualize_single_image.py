import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result

def get_coco_cls():
    return {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'airplane': 4,
        'bus': 5,
        'train': 6,
        'truck': 7,
        'boat': 8,
        'traffic light': 9,
        'fire hydrant': 10,
        'stop sign': 11,
        'parking meter': 12,
        'bench': 13,
        'bird': 14,
        'cat': 15,
        'dog': 16,
        'horse': 17,
        'sheep': 18,
        'cow': 19,
        'elephant': 20,
        'bear': 21,
        'zebra': 22,
        'giraffe': 23,
        'backpack': 24,
        'umbrella': 25,
        'handbag': 26,
        'tie': 27,
        'suitcase': 28,
        'frisbee': 29,
        'skis': 30,
        'snowboard': 31,
        'sports ball': 32,
        'kite': 33,
        'baseball bat': 34,
        'baseball glove': 35,
        'skateboard': 36,
        'surfboard': 37,
        'tennis racket': 38,
        'bottle': 39,
        'wine glass': 40,
        'cup': 41,
        'fork': 42,
        'knife': 43,
        'spoon': 44,
        'bowl': 45,
        'banana': 46,
        'apple': 47,
        'sandwich': 48,
        'orange': 49,
        'broccoli': 50,
        'carrot': 51,
        'hot dog': 52,
        'pizza': 53,
        'donut': 54,
        'cake': 55,
        'chair': 56,
        'couch': 57,
        'potted plant': 58,
        'bed': 59,
        'dining table': 60,
        'toilet': 61,
        'tv': 62,
        'laptop': 63,
        'mouse': 64,
        'remote': 65,
        'keyboard': 66,
        'cell phone': 67,
        'microwave': 68,
        'oven': 69,
        'toaster': 70,
        'sink': 71,
        'refrigerator': 72,
        'book': 73,
        'clock': 74,
        'vase': 75,
        'scissors': 76,
        'teddy bear': 77,
        'hair drier': 78,
        'toothbrush': 79
    }



# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model_path, class_list):

    # with open(class_list, 'r') as f:
    #     classes = load_classes(csv.reader(f, delimiter=','))
    classes = get_coco_cls()
    labels = {}
    for key, value in classes.items():
        labels[value] = key

    model = torch.load(model_path)

    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()

    for img_name in os.listdir(image_path)[4:5]:

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            print(image.shape, image_orig.shape, scale)
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]

                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                # draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imwrite('./result/output.png', image_orig)
            cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default='../../01_data/coco128_coco/images/train2017', help='Path to directory containing images')
    parser.add_argument('--model_path', default='../../02_models/fgai_trained_models/chapter06/13_pytorch-retinanet/model_final.pt', help='Path to model')
    parser.add_argument('--class_list', default='.', help='Path to CSV file listing class names (see README)')

    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
