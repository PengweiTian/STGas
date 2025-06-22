import numpy as np
import random


def apply_expand(imglist, tubes, expand_param, mean_values=None):
    # Tubes: dict of label -> list of tubes with tubes being <x1> <y1> <x2> <y2>
    out_imglist = imglist
    out_tubes = tubes
    expand_ratio = 1.0
    if random.random() < expand_param['expand_prob']:
        expand_ratio = random.uniform(1, expand_param['max_expand_ratio'])
        oh, ow = imglist[0].shape[:2]
        h = int(oh * expand_ratio)
        w = int(ow * expand_ratio)
        out_imglist = [np.zeros((h, w, 3), dtype=np.float32) for i in range(len(imglist))]
        h_off = int(np.floor(h - oh))
        w_off = int(np.floor(w - ow))
        if mean_values is not None:
            for i in range(len(imglist)):
                out_imglist[i] += np.array(mean_values).reshape(1, 1, 3)
        for i in range(len(imglist)):
            out_imglist[i][h_off:h_off + oh, w_off:w_off + ow, :] = imglist[i]
        # project boxes
        for ilabel in tubes:
            for itube in range(len(tubes[ilabel])):
                out_tubes[ilabel][itube] += np.array([[w_off, h_off, w_off, h_off]], dtype=np.float32)

    return out_imglist, out_tubes, expand_ratio


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)


def sample_cuboids(tubes, batch_samplers, imheight, imwidth):
    sampled_cuboids = []
    for batch_sampler in batch_samplers:
        max_trials = batch_sampler['max_trials']
        max_sample = batch_sampler['max_sample']
        itrial = 0
        isample = 0
        sampler = batch_sampler['sampler']

        min_scale = sampler['min_scale'] if 'min_scale' in sampler else 1
        max_scale = sampler['max_scale'] if 'max_scale' in sampler else 1
        min_aspect = sampler['min_aspect_ratio'] if 'min_aspect_ratio' in sampler else 1
        max_aspect = sampler['max_aspect_ratio'] if 'max_aspect_ratio' in sampler else 1

        while itrial < max_trials and isample < max_sample:
            # sample a normalized box
            scale = random.uniform(min_scale, max_scale)
            aspect = random.uniform(min_aspect, max_aspect)
            width = scale * np.sqrt(aspect)
            height = scale / np.sqrt(aspect)
            if width > 1 or height > 1:
                continue
            x = random.uniform(0, 1 - width)
            y = random.uniform(0, 1 - height)

            # rescale the box
            sampled_cuboid = np.array([x * imwidth, y * imheight, (x + width) * imwidth, (y + height) * imheight],
                                      dtype=np.float32)
            # check constraint
            itrial += 1
            if not 'sample_constraint' in batch_sampler:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

            constraints = batch_sampler['sample_constraint']
            ious = np.array([np.mean(iou2d(t, sampled_cuboid)) for t in sum(tubes.values(), [])])
            if ious.size == 0:  # empty gt
                isample += 1
                continue

            if 'min_jaccard_overlap' in constraints and ious.max() >= constraints['min_jaccard_overlap']:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

            if 'max_jaccard_overlap' in constraints and ious.min() >= constraints['max_jaccard_overlap']:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

    return sampled_cuboids


def crop_image(imglist, tubes, batch_samplers):
    candidate_cuboids = sample_cuboids(tubes, batch_samplers, imglist[0].shape[0], imglist[0].shape[1])
    crop_areas = [0.0, 0.0, 1.0, 1.0]
    if not candidate_cuboids:
        return imglist, tubes, crop_areas
    h = imglist[0].shape[0]
    w = imglist[0].shape[1]
    crop_cuboid = random.choice(candidate_cuboids)
    x1, y1, x2, y2 = map(int, crop_cuboid.tolist())

    for i in range(len(imglist)):
        imglist[i] = imglist[i][y1:y2 + 1, x1:x2 + 1, :]
    out_tubes = {}
    wi = x2 - x1
    hi = y2 - y1
    # if x1<0 or x2>w or y1<0 or y2>h:
    #    print('error:')
    #    print(x1,y1,x2,y2)
    #    print(imglist[0].shape)

    for ilabel in tubes:
        for itube in range(len(tubes[ilabel])):
            t = tubes[ilabel][itube]
            t -= np.array([[x1, y1, x1, y1]], dtype=np.float32)

            # check if valid
            cx = 0.5 * (t[:, 0] + t[:, 2])
            cy = 0.5 * (t[:, 1] + t[:, 3])

            if np.any(cx < 0) or np.any(cy < 0) or np.any(cx > wi) or np.any(cy > hi):
                continue

            if not ilabel in out_tubes:
                out_tubes[ilabel] = []

            # clip box
            t[:, 0] = np.maximum(0, t[:, 0])
            t[:, 1] = np.maximum(0, t[:, 1])
            t[:, 2] = np.minimum(wi, t[:, 2])
            t[:, 3] = np.minimum(hi, t[:, 3])

            out_tubes[ilabel].append(t)
    crop_areas = [x1 / w, y1 / h, x2 / w, y2 / h]
    return imglist, out_tubes, crop_areas
