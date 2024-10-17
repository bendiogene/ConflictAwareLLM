You can generate a new conf file as follows for a new experiment (starting from the old GPT2 template). 


Below is just an example, run the help again for an up to date version.

```
 python configs/generate_yml_config.py --help
usage: generate_yml_config.py [-h] [--model_name MODEL_NAME]
                              [--training_lr TRAINING_LR]
                              [--classification_lr CLASSIFICATION_LR]
                              [--http_proxy HTTP_PROXY]
                              [--https_proxy HTTPS_PROXY]
                              [--experiment_name EXPERIMENT_NAME]
                              [--cuda_visible_devices CUDA_VISIBLE_DEVICES]
                              [--CLASSIFICATION_device CLASSIFICATION_DEVICE]
                              [--max_length_FT MAX_LENGTH_FT]
                              [--max_length_EXTRACTION MAX_LENGTH_EXTRACTION]
                              [--max_length_CLASSIFICATION MAX_LENGTH_CLASSIFICATION]
                              [--max_length_LOCATION MAX_LENGTH_LOCATION]
                              [--sample_size_FT SAMPLE_SIZE_FT]
                              [--sample_size_EXTRACTION SAMPLE_SIZE_EXTRACTION]
                              [--sample_size_LOCATION SAMPLE_SIZE_LOCATION]
                              [--num_epochs_FT NUM_EPOCHS_FT]
                              [--num_epochs_CLASSIFICATION NUM_EPOCHS_CLASSIFICATION]
                              [--batch_size_FT BATCH_SIZE_FT]
                              [--batch_size_CLASSIFICATION BATCH_SIZE_CLASSIFICATION]
                              [--batch_size_EXTRACTION BATCH_SIZE_EXTRACTION]
                              [--batch_size_LOCATION BATCH_SIZE_LOCATION]
                              [--output_file OUTPUT_FILE]

Generate a YAML configuration file.

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name (default: gpt2small)
  --training_lr TRAINING_LR
                        Training learning rate (default: 0.0015)
  --classification_lr CLASSIFICATION_LR
                        Classification learning rate (default: 0.0001)
  --http_proxy HTTP_PROXY
                        HTTP proxy (default: http://localhost:3166)
  --https_proxy HTTPS_PROXY
                        HTTPS proxy (default: http://localhost:3166)
  --experiment_name EXPERIMENT_NAME
                        Experiment name (default: experiment1)
  --cuda_visible_devices CUDA_VISIBLE_DEVICES
                        CUDA visible devices (default: cuda:2)
  --CLASSIFICATION_device CLASSIFICATION_DEVICE
                        Device for classification (default: cuda:0)
  --max_length_FT MAX_LENGTH_FT
                        Max length for fine-tuning (default: 128)
  --max_length_EXTRACTION MAX_LENGTH_EXTRACTION
                        Max length for extraction (default: 128)
  --max_length_CLASSIFICATION MAX_LENGTH_CLASSIFICATION
                        Max length for classification (default: 128)
  --max_length_LOCATION MAX_LENGTH_LOCATION
                        Max length for location (default: 128)
  --sample_size_FT SAMPLE_SIZE_FT
                        Sample size for fine-tuning (default: 1000)
  --sample_size_EXTRACTION SAMPLE_SIZE_EXTRACTION
                        Sample size for extraction (default: 1000)
  --sample_size_LOCATION SAMPLE_SIZE_LOCATION
                        Sample size for location (default: 1000)
  --num_epochs_FT NUM_EPOCHS_FT
                        Number of epochs for fine-tuning (default: 5)
  --num_epochs_CLASSIFICATION NUM_EPOCHS_CLASSIFICATION
                        Number of epochs for classification (default: 3000)
  --batch_size_FT BATCH_SIZE_FT
                        Batch size for fine-tuning (default: 32)
  --batch_size_CLASSIFICATION BATCH_SIZE_CLASSIFICATION
                        Batch size for classification (default: 256)
  --batch_size_EXTRACTION BATCH_SIZE_EXTRACTION
                        Batch size for extraction (default: 1)
  --batch_size_LOCATION BATCH_SIZE_LOCATION
                        Batch size for location (default: 1)
  --output_file OUTPUT_FILE
                        Output YAML file name (default: new_config.yml)
```


```
python configs/generate_yml_config.py --model_name gpt2 --experiment_name experiment1
```