import torchvision.transforms as transforms

CIFAR10_ROOT = '/vulcanscratch/psando/cifar-10/'
CIFAR100_ROOT = '/vulcanscratch/psando/cifar-100'
SVHN_ROOT = '/vulcanscratch/psando/SVHN'
IMAGENET100_ROOT = '/vulcanscratch/psando/imagenet'

# POISON_SETUPS contains dictionaries containing the root, dataset name, and dataset type
# Available dataset_name: CIFAR10, CIFAR100, SVHN, IMAGENET100, IMAGENET2
# Available dataset_type: STANDARD, TAP, ULE, ULE_IMAGENET, NTGA, REM
# Details on dataset types:
#   TAP (Targeted Adversarial Poisoning): directory contains a data/ subdirectory with poisoned images
#   ULE (Unlearnable): directory contains perturbation.pt file
DATA_SETUPS = {
    'cifar10'         : {'root': CIFAR10_ROOT,
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'STANDARD'},
    'cifar100'        : {'root': CIFAR100_ROOT,
                         'dataset_name': 'CIFAR100',
                         'dataset_type': 'STANDARD'},
    'svhn'            : {'root': SVHN_ROOT,
                         'dataset_name': 'SVHN',
                         'dataset_type': 'STANDARD'},
    'imagenet100'     : {'root': IMAGENET100_ROOT,
                         'dataset_name': 'IMAGENET100',
                         'dataset_type': 'STANDARD'},
    'ntga'            : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/no_bound/ntga',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'NTGA'},
    'error-max'       : {'root':'/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/targeted_ResNet18_iter_250',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'untargeted-error-max': {'root':'/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/untargeted_ResNet18_iter_250',
                             'dataset_name': 'CIFAR10',
                             'dataset_type': 'TAP'},
    'error-min'       : {'root':'/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/unlearnable_samplewise',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'ULE'},
    'error-min-CW'    : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/unlearnable_classwise',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'ULE'},
    'robust-error-min': {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/linf-8-robust-error-min/rem-fin-def-noise.pkl',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'REM'},
    'ar'              : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/linf-8-cifar10-ar',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'l2-ar'           : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/l2/eps-1/mr-10-eps-1/',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'regions-4'       : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/regions-4',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'regions-16'      : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/regions-16',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'cwrandom'        : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/classwise_random_eps_8',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'l2-regions-4'    : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/l2/eps-1/l2-regions-4/',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'patches-4'       : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/patches-4x4',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'patches-8'       : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar10/linf/patches-8x8',
                         'dataset_name': 'CIFAR10',
                         'dataset_type': 'TAP'},
    'svhn-error-min-CW' : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/svhn/unlearnable_classwise',
                           'dataset_name': 'SVHN',
                           'dataset_type': 'ULE'},
    'svhn-error-min-SW' : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/svhn/unlearnable_samplewise',
                           'dataset_name': 'SVHN',
                           'dataset_type': 'ULE'},
    'svhn-l2-ar'         : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/svhn/svhn-mr-10-eps-1',
                            'dataset_name': 'SVHN',
                            'dataset_type': 'TAP'},
    'cifar100-error-min-CW' : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar100/unlearnable_classwise',
                               'dataset_name': 'CIFAR100',   
                               'dataset_type': 'ULE'},
    'cifar100-error-min-SW' : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar100/unlearnable_samplewise',
                               'dataset_name': 'CIFAR100',
                               'dataset_type': 'ULE'},
    'cifar100-l2-ar'        : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/cifar100/cifar100-mr-3-eps-1',
                               'dataset_name': 'CIFAR100',
                               'dataset_type': 'TAP'},
    'imagenet100-error-min-CW' : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/imagenet-100class/unlearnable_classwise',
                                  'dataset_name': 'IMAGENET100',
                                  'dataset_type': 'ULE_IMAGENET'},
    'imagenet100-error-min-SW' : {'root': '/fs/vulcan-projects/stereo-detection/tmp-poison/imagenet-100class/unlearnable_samplewise',
                                  'dataset_name': 'IMAGENET100',
                                  'dataset_type': 'ULE_IMAGENET'},
    'imagenet2-ntga'           : {'root': '/fs/vulcan-projects/stereo-detection/psando_poisons/paper/imagenet-2class/no_bound/ntga',
                                  'dataset_name': 'IMAGENET2',
                                  'dataset_type': 'NTGA'},
}

TRANSFORM_OPTIONS = {
    "CIFAR10": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(20),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "IMAGENET100": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},
}

NORMALIZE_CONSTANTS = {
    "CIFAR10": {'mean': [0.4914, 0.4822, 0.4465], 
                'std': [0.2023, 0.1994, 0.2010]},
    "CIFAR100": {'mean': [0.4914, 0.4822, 0.4465], 
                'std': [0.2023, 0.1994, 0.2010]},
    "SVHN": {'mean': [0.4914, 0.4822, 0.4465], 
                'std': [0.2023, 0.1994, 0.2010]},        
    "IMAGENET100": {'mean': [0.485, 0.456, 0.406], 
                    'std': [0.229, 0.224, 0.225]}
}

NUM_CLASSES = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "SVHN": 10,
    "IMAGENET100": 100,
}

LINEAR_CKPT_DIR = '/fs/vulcan-projects/stereo-detection/poisoning-defenses/linear'
MODEL_CKPT_DIR = '/fs/vulcan-projects/stereo-detection/poisoning-defenses/checkpoints/'
AR_AUG_COEFFS = 'ar-coeff-50_11-8.pt'