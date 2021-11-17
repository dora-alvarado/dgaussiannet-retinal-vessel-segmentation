import numpy as np
import cv2


def paste_imgs(lst_imgs, n_rows, n_cols, sep=5):
    n_imgs = len(lst_imgs)

    assert (n_imgs <= n_rows * n_cols)
    h, w = lst_imgs[0].shape[:2]

    new_im = np.ones(((h+sep)*n_rows+sep, (w+sep)*n_cols+sep,3))*255

    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            img = lst_imgs[k]
            if len(img.shape)==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            row = (h+sep)*i+sep
            col = (w+sep)*j+sep
            new_im[row:row+h, col:col+w] = img
            k+=1
            if k>=n_imgs:
                break
    if sep>0:
        return new_im[sep:-sep,sep:-sep]
    return new_im

def plot_errors(im_pred, im_gt, lineThickness=1):
    im_pred = im_pred>0.5
    m, n = im_gt.shape
    # output image
    new_im = np.ones((m, n, 3), dtype=np.uint8)*255#np.copy(img).astype(np.uint8)
    # colors for well predicted, false positives and false negative
    color_gt = (0, 0, 0)
    color_fp = (255, 0, 0)
    color_fn = (0, 0, 255)
    # well predicted
    tp = (im_pred != 0) & (im_gt != 0)
    idx = np.asarray(np.argwhere(tp), dtype=int)
    for y, x in idx:
        new_im[y,x, :]=color_gt#cv2.circle(new_im, (x, y), lineThickness // 2, color_gt, -1)
    # false positives
    fp = (im_pred !=0) & (im_gt==0)
    idx = np.asarray(np.argwhere(fp), dtype=int)
    for y, x in idx:
        new_im[y,x, :]=color_fp#cv2.circle(new_im, (x, y), lineThickness // 2, color_fp, -1)
    # false negatives
    fn = (im_pred == 0) & (im_gt != 0)
    idx = np.asarray(np.argwhere(fn), dtype=int)
    for y, x in idx:
        new_im[y,x, :]=color_fn#cv2.circle(new_im, (x, y), lineThickness // 2, color_fn, -1)
    return new_im
