
DEFAULT_HYPERPARAMETERS = {
    'id': '0',
    'num_classes': 6,
    'epochs': 3,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'num_hidden': 512,
    'last_layer_only': False,
    'model_name': "resnet18",
    'train_batch_size': 16,
    'test_batch_size': 16
}

HYPERPARAM_LIST = [DEFAULT_HYPERPARAMETERS,
                   #1
                   {'id': '1', 'num_hidden': 64, 'last_layer_only': True, 'learning_rate': 0.001, 'model_name': "resnet18",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   {'id': '2', 'num_hidden': 64, 'last_layer_only': True, 'learning_rate': 0.005, 'model_name': "resnet18",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   {'id': '3', 'num_hidden': 1024, 'last_layer_only': False, 'learning_rate': 0.001, 'model_name': "resnet18",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},

                    # Resnet 50
                   {'id': '4', 'num_hidden': 64, 'last_layer_only': True, 'learning_rate': 0.001, 'model_name': "resnet50",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   {'id': '5', 'num_hidden': 64, 'last_layer_only': True, 'learning_rate': 0.005, 'model_name': "resnet50",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   {'id': '6', 'num_hidden': 1024, 'last_layer_only': False, 'learning_rate': 0.001, 'model_name': "resnet50",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},

                   # Resnet101
                   {'id': '7', 'num_hidden': 64, 'last_layer_only': True, 'learning_rate': 0.001, 'model_name': "resnet101",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   {'id': '8', 'num_hidden': 64, 'last_layer_only': True, 'learning_rate': 0.005, 'model_name': "resnet101",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   {'id': '9', 'num_hidden': 1024, 'last_layer_only': False, 'learning_rate': 0.001, 'model_name': "resnet101",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},

                   #10
                   {'id': '10', 'num_hidden': 1024, 'last_layer_only': False, 'learning_rate': 0.01, 'model_name': "resnet18",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   #11
                   {'id': '11', 'num_hidden': 1024, 'last_layer_only': True, 'learning_rate': 0.01, 'model_name': "resnet18",
                    'train_batch_size': 16, 'subsample': 1, 'momentum': 0.9, 'epochs': 10},
                   ]

HYPERPARAM_DICT = {h['id']: dict(DEFAULT_HYPERPARAMETERS, **h) for i,h in enumerate(HYPERPARAM_LIST)}

# import json
# with open('../experiment_drivers/hyp_map_only18.jsonhyp_map_default.json', 'w') as f:
#     json.dump(HYPERPARAM_DICT, f)