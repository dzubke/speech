import torch
import argparse


def export_state_dict(model_in_path, params_out_path):
    model = torch.load(model_in_path, map_location=torch.device('cpu'))
    torch.save(model.state_dict(), params_out_path)

def main(model_path, params_path):
    
    export_state_dict(model_path, params_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("params_path")
    args = parser.parse_args()

    main(args.model_path, args.params_path)

