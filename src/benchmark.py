import os
import argparse
from collections import defaultdict
import subprocess
from omegaconf import OmegaConf


def main():
    # Create ArgumentParser object
    model_types = ["coref", "mei"]  ## Add / to this while passing arguments
    trained_ons = ["lf", "o"]  ## Add / to this while passing arguments
    run_files = {
        "coref": "configs/runs_coref.yaml",
        "mei": "configs/runs_mei.yaml",
    }
    script_paths = {"coref": "evaluate.baselines", "mei": "evaluate.meira"}
    setups = ["e2e"]
    modes = {"coref": ["coref_id", "coref_cm", "coref_fm"], "mei": ["static", "hybrid"]}
    datasets = "lf"

    for model_type in model_types:
        script_path = script_paths[model_type]
        for trained_on in trained_ons:
            for setup in setups:
                for mode in modes[model_type]:
                    if model_type == "coref":
                        all_runs = OmegaConf.load(run_files[model_type])[trained_on]
                    else:
                        all_runs = OmegaConf.load(run_files[model_type])[trained_on][
                            mode
                        ]
                    for run in all_runs:
                        model_base_path = all_runs[run]
                        # print(
                        #     f"Running: Model:{model_base_path}, Model Type:{model_type}, Train:{train}, Mode:{mode}"
                        # )
                        args_script = f'paths.base_addr="bench" paths.model_base_path="{model_base_path}" run="{run}" type="{model_type}/" trained_on="{trained_on}/" mode="{mode}/" setup="{setup}" datasets="{datasets}"'

                        try:
                            command = f"python -m {script_path} {args_script}"
                            print(f"Running: {command}")
                            result = subprocess.run(
                                command,
                                # stdout=subprocess.PIPE,
                                # stderr=subprocess.PIPE,
                                text=True,
                                shell=True,
                            )
                            if result.returncode != 0:
                                print(f"Error Output:{result.stderr}")
                                print(f"Error in: Config File:{command}, Run:{run}")
                        except FileNotFoundError:
                            print("Error: Python executable not found.")


if __name__ == "__main__":
    main()
