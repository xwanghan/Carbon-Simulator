import argparse
import logging
import os
import sys
import time
import torch_models
import matplotlib.pyplot as plt
import numpy as np

import ray
import utils.saving as saving
import yaml
from env_wrapper import RLlibEnvWrapper
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger, pretty_print

ray.init(log_to_driver=False)

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir", type=str, default='exp', help="Path to the directory for this run."
    )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration


def build_trainer(run_configuration):
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_configuration.get("trainer")

    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config)

    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    # Which policies to train
    if run_configuration["general"]["train_planner"] and not run_configuration["general"]["fix_mobile"]:
        policies_to_train = ["a", "p"]
    elif not run_configuration["general"]["train_planner"] and not run_configuration["general"]["fix_mobile"]:
        policies_to_train = ["a"]
    elif run_configuration["general"]["train_planner"] and run_configuration["general"]["fix_mobile"]:
        policies_to_train = ["p"]
    else:
        raise ValueError("must train one agent")

    # === Finalize and create ===
    trainer_config.update(
        {
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "a" if str(agent_id).isdigit() else "p",
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
                                          * trainer_config.get("num_envs_per_worker"),
        }
    )

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    ppo_trainer = PPOConfig().update_from_dict(trainer_config).build(env=RLlibEnvWrapper, logger_creator=logger_creator)

    return ppo_trainer

def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1.0.0. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_a_ok = saving.load_snapshot(
            trainer_obj, run_directory, load_latest=True
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )

    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only torch checkpoint weights
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents weights...")
            saving.load_model_weights(trainer_obj, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent weights.")

        starting_weights_path_planner = run_configuration["general"].get(
            "restore_weights_planner", ""
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner weights...")
            saving.load_model_weights(trainer_obj, starting_weights_path_planner)
        else:
            logger.info("Starting with fresh planner weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


def maybe_store_dense_log(
        trainer_obj, result_dict, dense_log_freq, dense_log_directory, trainer_step_last_ckpt
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        training_iteration = result_dict["training_iteration"]

        if training_iteration == 1 or training_iteration - trainer_step_last_ckpt >= dense_log_freq:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:06d}".format(result_dict["training_iteration"]),
            )
            trainer_step_last_ckpt = int(training_iteration)
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            saving.write_dense_logs(trainer_obj, log_dir)
            logger.info(">> Wrote dense logs to: %s", log_dir)

    return trainer_step_last_ckpt

def maybe_save(trainer_obj, result_dict, ckpt_freq, ckpt_directory, trainer_step_last_ckpt):
    training_iteration = result_dict["training_iteration"]

    # Check if saving this iteration
    if (
            result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if training_iteration - trainer_step_last_ckpt >= ckpt_freq:
                saving.save_snapshot(trainer_obj, ckpt_directory, suffix="")
                saving.save_model_weights(
                    trainer_obj, ckpt_directory, training_iteration, suffix="agent"
                )
                saving.save_model_weights(
                    trainer_obj, ckpt_directory, training_iteration, suffix="planner"
                )

                trainer_step_last_ckpt = int(training_iteration)

                logger.info("Checkpoint saved @ step %d", training_iteration)

    return trainer_step_last_ckpt

def plot_reward(run_directory, reward_a, reward_p):
    np_dir = run_directory + "/reward_a.npy"
    np.save(np_dir, np.array(reward_a))

    np_dir = run_directory + "/reward_p.npy"
    np.save(np_dir, np.array(reward_p))

    fig1 = plt.figure()
    plt.plot(range(len(reward_a)), reward_a)
    fig_dir = run_directory + "/reward_a.jpg"
    fig1.savefig(fig_dir)
    plt.close()

    fig2 = plt.figure()
    plt.plot(range(len(reward_a)), reward_p)
    fig_dir = run_directory + "/reward_p.jpg"
    fig2.savefig(fig_dir)
    plt.close()


if __name__ == "__main__":


    # ===================
    # === Start setup ===
    # ===================

    # Process the args
    run_dir, run_config = process_args()

    fh = logging.FileHandler(run_dir+"/train.log")
    logger.addHandler(fh)

    # Create a trainer object
    trainer = build_trainer(run_config)

    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(run_dir, run_config, trainer)

    # ======================
    # === Start training ===
    # ======================
    dense_log_frequency = run_config["general"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)
    step_last_log = 0

    reward_result_a, reward_result_p = [], []

    while num_parallel_episodes_done < run_config["general"]["episodes"]:

        # Training
        result = trainer.train()

        # === Counters++ ===
        num_parallel_episodes_done = result["episodes_total"]
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

        logger.info(
            "Iter %d: episodes this-iter %d total %d step -> %d/%d episodes done",
            curr_iter,
            result["episodes_this_iter"],
            global_step,
            num_parallel_episodes_done,
            run_config["general"]["episodes"],
        )

        if curr_iter == 1 or result["episodes_this_iter"] > 0:
            logger.info(pretty_print(result))

        reward_result_a.append(result.get('policy_reward_mean')["a"] if result.get('policy_reward_mean') else 0)
        reward_result_p.append(result.get('policy_reward_mean')["p"] if result.get('policy_reward_mean') else 0)
        plot_reward(run_dir, reward_result_a, reward_result_p)

        # === Dense logging ===
        step_last_log = maybe_store_dense_log(trainer, result, dense_log_frequency, dense_log_dir, step_last_log)

        # === Saving ===
        step_last_ckpt = maybe_save(
            trainer, result, ckpt_frequency, ckpt_dir, step_last_ckpt
        )

    # Finish up
    logger.info("Completing! Saving final snapshot...\n\n")
    saving.save_snapshot(trainer, ckpt_dir)
    saving.save_model_weights(trainer, ckpt_dir, global_step, suffix="agent")
    saving.save_model_weights(trainer, ckpt_dir, global_step, suffix="planner")
    logger.info("Final snapshot saved! All done.")

    ray.shutdown()  # shutdown Ray after use
