import yaml


with open("config.yaml", "r") as f:
    run_configuration = yaml.safe_load(f)

run_configuration
