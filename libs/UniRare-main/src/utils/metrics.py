import cv2
import random
import numpy as np
import skimage.morphology as morph
import torch
import torch.nn.functional as F

def compute_msr(salmap, targmap, distmap):
    """
    Calcule les ratios de saillance maximale (MSRt, MSRb).
    
    Args:
        target_saliency (torch.Tensor): Saillance de la cible.
        distractor_saliency (torch.Tensor): Saillance des distracteurs.
        background_saliency (torch.Tensor): Saillance de l'arrière-plan.
    
    Returns:
        tuple: (MSRt, MSRb).
    """

    msrt = MSR_targ(salmap,targmap,distmap)
    msrb = MSR_bg(salmap,targmap,distmap)
    return {'msrt':round(msrt,4), 'msrb':round(msrb,4)}


def MSR_targ(salmap, targmap, distmap, dilate=0, add_eps=False):
    if isinstance(salmap, str):
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE) # assume that this is a grayscale binary map with white for target and black for non-target
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE) # assume that this is a grayscale binary map with white for distractors and black for non-distractors

    if add_eps:
        randimg = [random.uniform(0, 1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_copy = targmap.copy()
    distmap_copy = distmap.copy()

    # dilate the target and distractor maps to allow for saliency bleed
    if dilate > 0:
        targmap_copy = morph.dilation(targmap_copy.astype(np.uint8), morph.disk(dilate))
        distmap_copy = morph.dilation(distmap_copy.astype(np.uint8), morph.disk(dilate))

    # convert the target and distractor masks into arrays with 0 and 1 for values
    targmap_normalized = targmap_copy / 255
    distmap_normalized = distmap_copy / 255
    salmap_normalized = salmap/255

    maxt = np.max(np.multiply(salmap_normalized, targmap_normalized))
    maxd = np.max(np.multiply(salmap_normalized, distmap_normalized))

    if maxd > 0:
        score = maxt/maxd
    else:
        score = -1

    return float( score )


def MSR_bg(salmap, targmap, distmap, dilate=0, add_eps=False):
    if isinstance(salmap, str):
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE) # a grayscale binary map with white for target and black for non-target
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE) # a grayscale binary map with white for distractors and black for non-distractors

    if add_eps:
        randimg = [random.uniform(0,1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_copy = targmap.copy()
    distmap_copy = distmap.copy()

    # dilate the target and distractor maps to allow for saliency bleed
    if dilate > 0:
        targmap_copy = morph.dilation(targmap_copy.astype(np.uint8), morph.disk(dilate))
        distmap_copy = morph.dilation(distmap_copy.astype(np.uint8), morph.disk(dilate))

    # convert the target and distractor masks into arrays with 0 and 1 for values
    targmap_normalized = targmap_copy / 255
    distmap_normalized = distmap_copy / 255
    salmap_normalized = salmap / 255
    # compute background mask from the target and distractor masks
    bgmap_normalized = 1 - np.logical_or(targmap_normalized > 0.5, distmap_normalized > 0.5)

    maxt = np.max(np.multiply(salmap_normalized, targmap_normalized))
    maxb = np.max(np.multiply(salmap_normalized, bgmap_normalized))

    if maxt > 0:
        score = maxb/maxt
    else:
        score = -1

    return float( score )




def NSS_score(saliencyMap, fixationMap):
    """
    Compute the Normalized Scanpath Saliency (NSS) score.

    Args:
        saliencyMap (numpy.ndarray): The saliency map.
        fixationMap (numpy.ndarray): The human fixation map (binary matrix).

    Returns:
        float: The NSS score.
    """
    map_resized = cv2.resize(saliencyMap, (fixationMap.shape[1], fixationMap.shape[0]))

    # normalize saliency map
    map_normalized = (map_resized - np.mean(map_resized)) / np.std(map_resized)

    # mean value at fixation locations
    score = np.mean(map_normalized[fixationMap.astype(bool)])

    return score



def CC_score(saliencyMap1, saliencyMap2):
    """
    Compute the Correlation Coefficient (CC) score between two saliency maps.

    Args:
        saliencyMap1 (numpy.ndarray): The first saliency map.
        saliencyMap2 (numpy.ndarray): The second saliency map.

    Returns:
        float: The CC score.
    """
    map1 = cv2.resize(saliencyMap1, (saliencyMap2.shape[1], saliencyMap2.shape[0])).astype(np.float64)
    map2 = saliencyMap2.astype(np.float64)

    # normalize both maps
    map1 = (map1 - np.mean(map1)) / np.std(map1)
    map2 = (map2 - np.mean(map2)) / np.std(map2)

    score = np.corrcoef(map1.flatten(), map2.flatten())[0, 1]

    return score


def KLdiv(saliencyMap, fixationMap):
    """
    Compute the KL-divergence between two maps.

    Args:
        saliencyMap (numpy.ndarray): The saliency map.
        fixationMap (numpy.ndarray): The human fixation map.

    Returns:
        float: The KL-divergence score.
    """
    map1 = cv2.resize(saliencyMap, (fixationMap.shape[1], fixationMap.shape[0])).astype(np.float64)
    map2 = fixationMap.astype(np.float64)

    # make sure map1 and map2 sum to 1
    if np.any(map1):
        map1 = map1 / np.sum(map1)
    if np.any(map2):
        map2 = map2 / np.sum(map2)

    # compute KL-divergence
    score = np.sum(map2 * np.log(np.finfo(float).eps + map2 / (map1 + np.finfo(float).eps)))
    return score


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


def AUC_Judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = cv2.resize(s_map, (gt.shape[1], gt.shape[0])).astype(np.float64)

    s_map = normalize_map(s_map)
    assert np.max(gt) <= 1.0, 'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) <= 1.0, 'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    return np.trapz(np.array(tp_list), np.array(fp_list))


def AUC_shuffled(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)



def SIM(saliencyMap1, saliencyMap2):
    """
    Compute the similarity score between two saliency maps.

    Args:
        saliencyMap1 (numpy.ndarray): The first saliency map.
        saliencyMap2 (numpy.ndarray): The second saliency map.
        toPlot (int, optional): If 1, displays output of similarity computation as well as both maps. Defaults to 0.

    Returns:
        float: The similarity score.
    """
    map1 = cv2.resize(saliencyMap1, (saliencyMap2.shape[1], saliencyMap2.shape[0])).astype(np.float64)
    map2 = saliencyMap2.astype(np.float64)

    if np.any(map1):
        map1 = (map1 - np.min(map1)) / (np.max(map1) - np.min(map1))
        map1 = map1 / np.sum(map1)

    if np.any(map2):
        map2 = (map2 - np.min(map2)) / (np.max(map2) - np.min(map2))
        map2 = map2 / np.sum(map2)

    if np.isnan(map1).all() or np.isnan(map2).all():
        return float('nan')

    diff = np.minimum(map1, map2)
    score = np.sum(diff)



    return score










import torch as t
import torch.nn as nn

def loss_KLdiv(pred_map, gt_map):
    eps = 2.2204e-16
    pred_map = pred_map/t.sum(pred_map)
    gt_map = gt_map/t.sum(gt_map)
    div = t.sum(t.mul(gt_map, t.log(eps + t.div(gt_map,pred_map+eps))))
    return div 
        
    
def loss_CC(pred_map,gt_map):
    gt_map_ = (gt_map - t.mean(gt_map))
    pred_map_ = (pred_map - t.mean(pred_map))
    cc = t.sum(t.mul(gt_map_,pred_map_))/t.sqrt(t.sum(t.mul(gt_map_,gt_map_))*t.sum(t.mul(pred_map_,pred_map_)))
    return cc


def loss_similarity(pred_map,gt_map):
    gt_map = (gt_map - t.min(gt_map))/(t.max(gt_map)-t.min(gt_map))
    gt_map = gt_map/t.sum(gt_map)
    
    pred_map = (pred_map - t.min(pred_map))/(t.max(pred_map)-t.min(pred_map))
    pred_map = pred_map/t.sum(pred_map)
    
    diff = t.min(gt_map,pred_map)
    score = t.sum(diff)
    
    return score
    
    
def loss_NSS(pred_map,fix_map):
    '''ground truth here is fixation map'''

    pred_map_ = (pred_map - t.mean(pred_map))/t.std(pred_map)
    mask = fix_map.gt(0)
    sco                                                                               