import yaml
import argparse
import sys


from pathlib import Path
# Path to the current script
current_script = Path(__file__).resolve()
current_script_path = current_script.parent


def generate_yaml(model_name, experiment_name, training_lr, classification_lr, http_proxy, https_proxy, cuda_visible_devices,
                  CLASSIFICATION_device, max_length_FT, max_length_EXTRACTION, max_length_CLASSIFICATION, max_length_LOCATION,
                  sample_size_FT, sample_size_EXTRACTION, sample_size_LOCATION, num_epochs_FT, num_epochs_CLASSIFICATION,
                  batch_size_FT, batch_size_CLASSIFICATION, batch_size_EXTRACTION, batch_size_LOCATION, output_file):
    original_yml = {
        'GENERAL': {
            'experiment_name': experiment_name,
            'model_name': model_name
        },
        'DEVICE_CONFIG': {
            'cuda_device_order': 'PCI_BUS_ID',
            'cuda_visible_devices': cuda_visible_devices,
            'proxies': {
                'http': http_proxy,
                'https': https_proxy
            }
        },
        'FT_CONFIG': {
            'models': [model_name],
            'conditional_generation_mode': ['focus_on_answer'],
            'dataset_file': 'dataset/facts/multi_counterfact.json',
            'save_dataset_path': f'experiments/{model_name}_{experiment_name}/model',
            'logging_dir': f'experiments/{model_name}_{experiment_name}/logs',
            'max_length': max_length_FT,
            'sample_size': sample_size_FT,
            'training_lr': training_lr,
            'num_epochs': num_epochs_FT,
            'batch_size': batch_size_FT,
            'eval_every_n_epochs': 1,
            'custom_metrics': False,
            'historical_data': True,
            'historical_file_path': f'experiments/{model_name}_{experiment_name}/history/historical_data.pkl'
        },
        'EXTRACTION_CONFIG': {
            'models': [model_name],
            'pretrained_model_dir': f'experiments/{model_name}_{experiment_name}/model',
            'unknown': True,
            'pretrained': True,
            'transformations': ['avg', 'std'],
            'dataset_file': 'dataset/facts/multi_counterfact.json',
            'dataset_file_unknown': 'dataset/facts/multi_counterfact_unknown.json',
            'save_dataset_path_familiar': f'experiments/{model_name}_{experiment_name}/classification/dataset/first_1000_familiar.pkl',
            'save_dataset_path_unknown': f'experiments/{model_name}_{experiment_name}/classification/dataset/first_1000_unknown.pkl',
            'save_dataset_path': f'experiments/{model_name}_{experiment_name}/classification/dataset/first_1000.pkl',
            'save_stats_path': f'experiments/{model_name}_{experiment_name}/extraction/distribution_data.pkl',
            'max_length': max_length_EXTRACTION,
            'sample_size': sample_size_EXTRACTION,
            'batch_size': batch_size_EXTRACTION,
            'norm': 'std',
            'save_stats': True
        },
        'CLASSIFICATION_CONFIG': {
            'unknown': True,
            'pretrained': True,
            'dataset_path': f'experiments/{model_name}_{experiment_name}/classification/dataset',
            'split': 'first_1000',
            'log_dir': f'experiments/{model_name}_{experiment_name}/classification/logs',
            'max_length': max_length_CLASSIFICATION,
            'batch_size': batch_size_CLASSIFICATION,
            'lr': classification_lr,
            'device': CLASSIFICATION_device,
            'n_classes':3,
            'layers': None,
            'n_epochs': num_epochs_CLASSIFICATION
        },
        'LOCATION_CONFIG': {
            'models': [model_name],
            'pretrained_model_dir': f'experiments/{model_name}_{experiment_name}/model',
            'dataset_file': 'dataset/facts/multi_counterfact.json',
            'dataset_file_unknown': 'dataset/facts/multi_counterfact_unknown.json',
            'save_dataset_path': f'experiments/{model_name}_{experiment_name}/location/location_data.pkl',
            'max_length': max_length_LOCATION,
            'sample_size': sample_size_LOCATION,
            'batch_size': batch_size_LOCATION,  
            'kn_location': False
        },
        'HISTORICAL_FEATURES_CONFIG': {
            'patterns' : [
                    r'\b\w+(?:\.attn\.c_attn)\b',
                    r'\b\w+(?:\.attn\.c_proj)\b',
                    r'\b\w+(?:\.mlp\.c_fc)\b',
                    r'\b\w+(?:\.mlp\.c_proj)\b'
                    ]
        }           
    }
    if output_file:
        pass
        
    else:
        output_file_name = f'{model_name}-{experiment_name}.yml'
        file_path = current_script_path / output_file_name
        with open(file_path, 'w') as file:
            yaml.dump(original_yml, file, sort_keys=False)




# Using argparse to get inputs from command line
parser = argparse.ArgumentParser(description='Generate a YAML configuration file. Regex expressions MUST be done manually after you generate the template.')

parser.add_argument('--model_name', type=str, default='gpt2', 
                    help='Model name (default: %(default)s)')
parser.add_argument('--experiment_name', type=str, default='experiment1', 
                    help='Experiment name (default: %(default)s)')
parser.add_argument('--training_lr', type=float, default=0.0015, 
                    help='Training learning rate (default: %(default)s)')
parser.add_argument('--classification_lr', type=float, default=0.0001, 
                    help='Classification learning rate (default: %(default)s)')
parser.add_argument('--http_proxy', type=str, default='http://localhost:3166', 
                    help='HTTP proxy (default: %(default)s)')
parser.add_argument('--https_proxy', type=str, default='http://localhost:3166', 
                    help='HTTPS proxy (default: %(default)s)')
parser.add_argument('--cuda_visible_devices', type=str, default='2', 
                    help='CUDA visible devices (default: %(default)s)')
parser.add_argument('--CLASSIFICATION_device', type=str, default='0', 
                    help='Device for classification (default: %(default)s)')
parser.add_argument('--max_length_FT', type=int, default=128, 
                    help='Max length for fine-tuning (default: %(default)s)')
parser.add_argument('--max_length_EXTRACTION', type=int, default=128, 
                    help='Max length for extraction (default: %(default)s)')
parser.add_argument('--max_length_CLASSIFICATION', type=int, default=128, 
                    help='Max length for classification (default: %(default)s)')
parser.add_argument('--max_length_LOCATION', type=int, default=128, 
                    help='Max length for location (default: %(default)s)')
parser.add_argument('--sample_size_FT', type=int, default=1000, 
                    help='Sample size for fine-tuning (default: %(default)s)')
parser.add_argument('--sample_size_EXTRACTION', type=int, default=1000, 
                    help='Sample size for extraction (default: %(default)s)')
parser.add_argument('--sample_size_LOCATION', type=int, default=1000, 
                    help='Sample size for location (default: %(default)s)')
parser.add_argument('--num_epochs_FT', type=int, default=5, 
                    help='Number of epochs for fine-tuning (default: %(default)s)')
parser.add_argument('--num_epochs_CLASSIFICATION', type=int, default=3000, 
                    help='Number of epochs for classification (default: %(default)s)')
parser.add_argument('--batch_size_FT', type=int, default=32, 
                    help='Batch size for fine-tuning (default: %(default)s)')
parser.add_argument('--batch_size_CLASSIFICATION', type=int, default=256, 
                    help='Batch size for classification (default: %(default)s)')
parser.add_argument('--batch_size_EXTRACTION', type=int, default=1, 
                    help='Batch size for extraction (default: %(default)s)')
parser.add_argument('--batch_size_LOCATION', type=int, default=1, 
                    help='Batch size for location (default: %(default)s)')
parser.add_argument('--output_file', type=str,
                    help='Output YAML file name. Default is model_name_experiment_name')

args = parser.parse_args()

if len(sys.argv) == 1:
    print("This script generates a new YAML configuration file based on given parameters.")
    print("Please provide arguments to customize the configuration, or use --help to see available options.")
    print("Note that Regex expressions MUST be done manually after you generate the template")
    sys.exit(1)  # Exit the script without error.

# Generate the YAML file with the parameters received from command line or default
args = parser.parse_args()

generate_yaml(
    model_name=args.model_name,
    experiment_name=args.experiment_name,
    training_lr=args.training_lr,
    classification_lr=args.classification_lr,
    http_proxy=args.http_proxy,
    https_proxy=args.https_proxy,
    cuda_visible_devices=args.cuda_visible_devices,
    CLASSIFICATION_device=args.CLASSIFICATION_device,
    max_length_FT=args.max_length_FT,
    max_length_EXTRACTION=args.max_length_EXTRACTION,
    max_length_CLASSIFICATION=args.max_length_CLASSIFICATION,
    max_length_LOCATION=args.max_length_LOCATION,
    sample_size_FT=args.sample_size_FT,
    sample_size_EXTRACTION=args.sample_size_EXTRACTION,
    sample_size_LOCATION=args.sample_size_LOCATION,
    num_epochs_FT=args.num_epochs_FT,
    num_epochs_CLASSIFICATION=args.num_epochs_CLASSIFICATION,
    batch_size_FT=args.batch_size_FT,
    batch_size_CLASSIFICATION=args.batch_size_CLASSIFICATION,
    batch_size_EXTRACTION=args.batch_size_EXTRACTION,
    batch_size_LOCATION=args.batch_size_LOCATION,
    output_file=args.output_file
)
