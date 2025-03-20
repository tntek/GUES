from PIL import Image, ImageFile
import random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import os.path as osp
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """
    ##### revised by luo
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # train_transforms = T.Compose([
        #         T.RandomResizedCrop(224, scale=(0.2, 1.0)),
        #         T.RandomApply(
        #             [T.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        #             p=0.8,  # not strengthened
        #         ),
        #         T.RandomGrayscale(p=0.2),
        #         T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        #         T.RandomHorizontalFlip(),
        #         T.ToTensor(),
        #         # normalize,
        #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     ])

        img_path, pid, camid,trackid, idx = self.dataset[index]
        if isinstance(img_path,tuple):
            all_imgs = []
            all_imgs_path = []
            for i in range(len(img_path)):
                i_path = img_path[i]
                i_img = read_image(i_path)
                # if self.transform is not None:
                #     if i == 0:
                #         i_img = train_transforms(i_img)
                #     else:
                #         i_img = self.transform(i_img)
                if self.transform is not None:
                    i_img = self.transform(i_img)
                all_imgs.append(i_img)
                all_imgs_path.append(i_path)
                # all_imgs_path.append(i_path.split('/')[-1])
            img = tuple(all_imgs)

            # print('data base pid ',pid)
            if isinstance(pid, tuple):
                if isinstance(idx, tuple):
                    return img + pid + (camid, trackid)+ tuple(all_imgs_path)+ idx
                else:
                    return img + pid + (camid, trackid, tuple(all_imgs_path), idx)
            else:
                return img + (pid, camid, trackid, tuple(all_imgs_path), idx)
        else:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
           
            return img, pid,camid, trackid, img_path,idx

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x