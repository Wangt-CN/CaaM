"""
Borrow from Rebias code
"""
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

CLASS_TO_INDEX = {'n01641577': 2, 'n01644373': 2, 'n01644900': 2, 'n01664065': 3, 'n01665541': 3,
                  'n01667114': 3, 'n01667778': 3, 'n01669191': 3, 'n01819313': 4, 'n01820546': 4,
                  'n01833805': 4, 'n01843383': 4, 'n01847000': 4, 'n01978287': 7, 'n01978455': 7,
                  'n01980166': 7, 'n01981276': 7, 'n02085620': 0, 'n02099601': 0, 'n02106550': 0,
                  'n02106662': 0, 'n02110958': 0, 'n02123045': 1, 'n02123159': 1, 'n02123394': 1,
                  'n02123597': 1, 'n02124075': 1, 'n02174001': 8, 'n02177972': 8, 'n02190166': 8,
                  'n02206856': 8, 'n02219486': 8, 'n02486410': 5, 'n02487347': 5, 'n02488291': 5,
                  'n02488702': 5, 'n02492035': 5, 'n02607072': 6, 'n02640242': 6, 'n02641379': 6,
                  'n02643566': 6, 'n02655020': 6}


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, data='ImageNet'):
    # dog, cat, frog, turtle, bird, monkey, fish, crab, insect
    RESTRICTED_RANGES = [(151, 254), (281, 285), (30, 32), (33, 37), (89, 97),
                         (372, 378), (393, 397), (118, 121), (306, 310)]
    range_sets = [set(range(s, e + 1)) for s, e in RESTRICTED_RANGES]
    class_to_idx_ = {}

    if data == 'ImageNet-A':
        for class_name, idx in class_to_idx.items():
            try:
                class_to_idx_[class_name] = CLASS_TO_INDEX[class_name]
            except Exception:
                pass
    elif data == 'ImageNet-C':
        # TODO
        pass
    else:  # ImageNet
        for class_name, idx in class_to_idx.items():
            for new_idx, range_set in enumerate(range_sets):
                if idx in range_set:
                    if new_idx == 0:  # classes that overlap with ImageNet-A
                        if idx in [151, 207, 234, 235, 254]:
                            class_to_idx_[class_name] = new_idx
                    elif new_idx == 4:
                        if idx in [89, 90, 94, 96, 97]:
                            class_to_idx_[class_name] = new_idx
                    elif new_idx == 5:
                        if idx in [372, 373, 374, 375, 378]:
                            class_to_idx_[class_name] = new_idx
                    else:
                        class_to_idx_[class_name] = new_idx
    images = []
    dir = os.path.expanduser(dir)
    a = sorted(class_to_idx_.keys())
    for target in a:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx_[target])
                    images.append(item)

    return images, class_to_idx_


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def make_env(imgs, class_to_idx_, n_env, env_type, pre_split=None):

    sample_num = len(imgs)
    sample_env = sample_num // n_env
    imgs_env = []


    if env_type == 'semi-auto':
        pre_cluster = torch.load('pre_cluster_results/cluster_label_1.pth')

        sort_zip = sorted(zip(imgs, pre_cluster), key=lambda x: x[1])
        imgs, pre_cluster = [list(x) for x in zip(*sort_zip)]  # sort by the cluster
        for env_idx in range(n_env):
            start_idx = env_idx * sample_env
            end_idx = (env_idx + 1) * sample_env if env_idx != n_env - 1 else sample_num
            imgs_env.append(imgs[start_idx:end_idx])


    elif env_type == 'random':
        import random
        random.shuffle(imgs)
        for env_idx in range(n_env):
            start_idx = env_idx * sample_env
            end_idx = (env_idx + 1) * sample_env if env_idx != n_env - 1 else sample_num
            imgs_env.append(imgs[start_idx:end_idx])


    elif env_type in ['auto-baseline', 'auto-iter', 'auto-iter-cluster']:
        ## use a reference model to make the env split
        ### initialize a split distribution

        assert pre_split is not None
        pre_cluster = torch.max(pre_split,1)[1]
        sort_zip = sorted(zip(imgs, pre_cluster), key=lambda x: x[1])
        imgs, pre_cluster = [list(x) for x in zip(*sort_zip)]  # sort by the cluster
        last_idx = 0
        for env_idx in range(n_env):
            sample_env = pre_cluster.count(env_idx)
            start_idx = last_idx
            end_idx = last_idx + sample_env
            imgs_env.append(imgs[start_idx:end_idx])
            last_idx = end_idx

    return imgs_env


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader,
                 train=True, val_data='ImageNet', soft_split=None):
        classes, class_to_idx = find_classes(root)
        imgs, class_to_idx_ = make_dataset(root, class_to_idx, val_data)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                   "Supported image extensions are: " + ",".join(
                       IMG_EXTENSIONS)))
        self.root = root
        self.dataset = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx_
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.val_data = val_data
        self.clusters = []
        for i in range(3):
            self.clusters.append(torch.load('clusters/cluster_label_{}.pth'.format(i+1)))
        self.soft_split = soft_split

    def __getitem__(self, index):
        path, target = self.dataset[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train and self.val_data == 'ImageNet':
            bias_target = [self.clusters[0][index],
                           self.clusters[1][index],
                           self.clusters[2][index]]
            return img, target, bias_target

        elif self.train and self.val_data == 'pre_dataloader':
            return img, target, index
        else:
            return img, target, target

    def __len__(self):
        return len(self.dataset)



### use to get the dataloader for each env split
class ImageFolder_env(torch.utils.data.Dataset):
    def __init__(self, imgs, root, transform=None, target_transform=None, loader=pil_loader,
                 train=True, val_data='ImageNet'):

        self.root = root
        self.dataset = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train
        self.val_data = val_data

    def __getitem__(self, index):
        path, target = self.dataset[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, target


    def __len__(self):
        return len(self.dataset)



def get_imagenet_dataloader(root, batch_size, train=True, num_workers=8,
                            load_size=256, image_size=224, val_data='ImageNet'):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    dataset = ImageFolder(root, transform=transform, train=train, val_data=val_data)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=train,
                                         num_workers=num_workers,
                                         pin_memory=True)

    return dataloader



def get_imagenet_dataloader_env(config, root, batch_size, train=True, num_workers=8,
                            load_size=256, image_size=224, val_data='ImageNet', pre_split=None):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    ### construct the split
    n_env = config['variance_opt']['n_env']
    classes, class_to_idx = find_classes(root)
    imgs, class_to_idx_ = make_dataset(root, class_to_idx, val_data)
    imgs_split = make_env(imgs, class_to_idx_, n_env, config['variance_opt']['env_type'], pre_split=pre_split)

    training_dataset = []
    training_dataset_all = ImageFolder(root, transform=transform, train=train, val_data=val_data)


    for env_idx in range(n_env):
        training_dataset.append(
            ImageFolder_env(imgs_split[env_idx], root, transform=transform, train=train, val_data=val_data))
    training_loader = []
    training_loader.append(
        torch.utils.data.DataLoader(CycleConcatDataset(*training_dataset), shuffle=train, num_workers=num_workers,
                   batch_size=batch_size, pin_memory=True))

    if config['variance_opt']['mode'] == 'ours':
        if config['variance_opt']['erm_flag']:
            training_dataset_erm = training_dataset_all
            training_loader.append(DataLoader(training_dataset_erm, shuffle=train, num_workers=num_workers,
                                              batch_size=n_env * batch_size, pin_memory=True))

    return training_loader



def get_pre_dataloader(config, root, batch_size, train=True, num_workers=8,
                            load_size=256, image_size=224, val_data='ImageNet', n_env=1):

    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    #dataset = ImageFolder(root, transform=transform, train=train, val_data=val_data)
    ### construct the split
    n_env = config['variance_opt']['n_env']
    classes, class_to_idx = find_classes(root)
    imgs, class_to_idx_ = make_dataset(root, class_to_idx, val_data)

    soft_split_init = torch.randn((len(imgs), n_env), device="cuda")
    soft_split_init = torch.nn.Parameter(soft_split_init)
    optimizer = torch.optim.SGD([soft_split_init], lr=0.1, momentum=0.9, weight_decay=0)
    # optimizer = torch.optim.Adam([soft_split_init], lr=0.01, weight_decay=0)
    optimizer.zero_grad()
    optimizer.step()
    pre_scheduler = MultiStepLR(optimizer, [30], gamma=0.1, last_epoch=-1)

    dataset = ImageFolder(root, transform=transform, train=train, val_data='pre_dataloader', soft_split=soft_split_init)
    training_loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=train,
                                         num_workers=num_workers,
                                         pin_memory=True)

    return training_loader, optimizer, pre_scheduler


def get_bias_dataloader(root, batch_size, train=True, num_workers=8, load_size=256, image_size=224, val_data='ImageNet'):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(load_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    dataset = ImageFolder(root, transform=transform, train=train, val_data='imagenet')
    training_loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=train,
                                         num_workers=num_workers,
                                         pin_memory=True)
    return training_loader

class CycleConcatDataset(Dataset):
    '''Dataset wrapping multiple train datasets
    Parameters
    ----------
    *datasets : sequence of torch.utils.data.Dataset
        Datasets to be concatenated and cycled
    '''
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        result = []
        for dataset in self.datasets:
            cycled_i = i % len(dataset)
            result.append(dataset[cycled_i])

        return tuple(result)

    def __len__(self):
        return max(len(d) for d in self.datasets)