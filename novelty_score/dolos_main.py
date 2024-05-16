from pathlib import Path

from utils.parse_arguments import parse_dolos_arguments
from utils.constants import (
    SOLUTION_TEMP_DIR,
)
from utils.save_results import save_dolos_results
from utils.clean_up import clean_up
from utils.dolos import create_solutions_tree, run_dolos_analysis


def main(input_path: Path, result_dir: Path) -> None:
    """Main function to orchestrate the Dolos job."""
    try:
        create_solutions_tree(input_path)
        results = run_dolos_analysis(list(SOLUTION_TEMP_DIR.iterdir()))
        save_dolos_results(results, result_dir, f"DolosMatch_{input_path.stem}")
    finally:
        clean_up(SOLUTION_TEMP_DIR)


if __name__ == "__main__":
    args = parse_dolos_arguments()
    main(Path(args.input_path), Path(args.result_dir))
