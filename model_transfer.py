import argparse
import speech
import torch





def print_layers(model_path, model_or_dict: bool = 'dict'):

    if model_or_dict == 'dict':
        model_dct= speech.load_pretrained(model_path)
        print(f"type: {model_dct['state_dict'].__dir__()}")
        print(f"type: {type(model_dct)}")
        print(f"dict keys: +{[names for names in model_dct.keys()]}")
        print(f"version: {model_dct['version']}")
        print(f"hidden_size: {model_dct['hidden_size']}")
        print(f"hidden_layers: {model_dct['hidden_layers']}")
        print(f"rnn_type: {model_dct['rnn_type']}")
        print(f"audio_conf: {model_dct['audio_conf']}")
        print(f"state dict layers: {[layers for layers in model_dct['state_dict'].keys()]}")
        # 4 CNN layers, 5 LSTM layers

        #pp_dict(model_dct)


        for name, param in model_dct['state_dict'].items():
            #print('=====')
            print('name: ', name)
            #print(type(param))
            print('param.shape: ', param.shape)
            #print('param.requires_grad: ', param.requires_grad)

    elif model_or_dict == 'model':
        model, _ = speech.load(model_path)
        #print(f"summary {model.summary}")
        print(f"dir: {model.__dir__()}")
        print(f"state_dict:")

        for name, param in model.state_dict().items():
            #print('=====')
            print('name: ', name)
            #print(type(param))
            print('param.shape: ', param.shape)
            #print('param.requires_grad: ', param.requires_grad)



def pp_dict(d, indent=0):
    """pretty prints the keys in a dictionary
    """
    for key in d.keys():
        print('\t' * indent + str(key))
        if isinstance(d[key], dict):
            pp_dict(d[key], indent+1)




if __name__ == "__main__":
    ### format of script command
    # python model_transfer.py <path_to_model> --model_or_dict
    parser = argparse.ArgumentParser(
            description="View and transfer layers of existing models.")
    parser.add_argument("model",
        help="Path to the pytorch model.")

    parser.add_argument("--model_or_dict",
        help="boolean flag that tells if the input path leads to a model object or dictinonary containing the state_dict.")

    args = parser.parse_args()

    print_layers(args.model, args.model_or_dict)