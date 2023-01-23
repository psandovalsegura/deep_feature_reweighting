CIFAR10_ROOT = '/vulcanscratch/psando/cifar-10/'
POISON_ROOTS = {
    'ntga'            : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/no_bound/ntga',
    'error-max'       : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/targeted_ResNet18_iter_250',
    'untargeted-error-max': '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/untargeted_ResNet18_iter_250',
    'error-min'       : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/unlearnable_samplewise',
    'robust-error-min': '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/linf-8-robust-error-min/rem-fin-def-noise.pkl',
    'ar'              : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/linf-8-cifar10-ar',
    'l2-ar'           : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/l2/eps-1/mr-10-eps-1/',
    'regions-4'       : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/regions-4',
    'regions-16'      : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/regions-16',
    'cwrandom'        : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/classwise_random_eps_8',
    'l2-regions-4'    : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/l2/eps-1/l2-regions-4/',
    'patches-4'       : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/patches-4x4',
    'patches-8'       : '/fs/vulcan-projects/robust_geometry_vsingla/psando_poisons/paper/cifar10/linf/patches-8x8'
}
TAP_FORMAT_POISONS = ['error-max', 'untargeted-error-max', 'ar', 'l2-ar', 'regions-4', 'regions-16', 'cwrandom', 'patches-4', 'patches-8', 'l2-regions-4'] # denotes which poisons are stored in the Adversarial Poisoning format
LINEAR_CKPT_DIR = '/fs/vulcan-projects/stereo-detection/poisoning-defenses/linear'
MODEL_CKPT_DIR = '/fs/vulcan-projects/stereo-detection/poisoning-defenses/checkpoints/'
AR_AUG_COEFFS = 'ar-coeff-50_11-8.pt'