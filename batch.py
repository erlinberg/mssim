import argparse
import logging
import os

from mssim.batching.batcher import MSSIMJobSubmitter

parser = argparse.ArgumentParser(description="MSSIM Batch Submitter")
parser.add_argument(
    "--config", type=str, default="settings.json", help="Path to config file"
)
parser.add_argument(
    "--parallel", action="store_true", help="Submit jobs in parallel to SLURM"
)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    job_submitter = MSSIMJobSubmitter(config_path=args.config)
    task_id_env = os.environ.get("SLURM_ARRAY_TASK_ID")

    if args.parallel and task_id_env is None:
        logger.info("Submitting batches to SLURM...")
        job_submitter.submit_parallel()
    elif args.parallel and task_id_env is not None:
        logger.info("Running for Task ID %s", task_id_env)
        job_submitter.run_task(task_id=int(task_id_env))
    else:
        logger.info("Submitting batches sequentially...")
        job_submitter.submit_sequential()
