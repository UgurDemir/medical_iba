MODEL_CONFIGS = {
    'covid1': {
        'model': {'arch': 'nest50', 'nclass':1, 'in_channel':1, 'b2m':True, 'istrained':False, 'out':None},
        'weight': '_models/19c/19c_resnest50_best_v_accuracy.pth.tar',
        'estim': '_models/19c/estim_19c_layer2.pth',
        'layer': 'layer2',
    }
}