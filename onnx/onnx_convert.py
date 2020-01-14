import torch
import argparse
import pickle


def onnx_export(input_path, output_path):
    torch_model = torch.load(input_path, map_location=torch.device('cpu'))
    dummy_path = '/Users/dustin/CS/consulting/firstlayerai/phoneme_classification/src/awni_speech/speech/dummy_batch'
    with open(dummy_path, 'rb') as fid:
        dummy_input = pickle.load(fid) 
    #dummy_tensor = (torch.from_numpy(dummy_input[0][0]).float().to('cpu'), [27])
    #dummy_tensor = [torch.FloatTensor(1, 200, 161), torch.IntTensor(1, 41)]
    dummy_tensor = [torch.randn(1, 200, 161, device='cpu'), torch.randn(1,10, device='cpu')]
    onnx_model = torch.onnx.export(torch_model, dummy_tensor, output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Eval a speech model.")

    parser.add_argument("input_model",
        help="A path to a stored model.")
    
    parser.add_argument("output_model",
        help="A path to a stored model.")
    args = parser.parse_args()

    onnx_export(args.input_model, args.output_model)