import argparse
import sys
from pathlib import Path
from speechlab import __version__
from speechlab.dataset import build_dataset, MANIFEST_PATH
from speechlab.features import extract_features
from speechlab.analysis import run_analysis
from speechlab.evaluation import run_evaluation

def main():
    parser = argparse.ArgumentParser(prog="speechlab", description="SpeechLab CLI for dataset building, feature extraction, analysis, and evaluation.")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-dataset command
    parser_build = subparsers.add_parser("build_dataset", help="Download/create the contrastive speech dataset.")
    parser_build.add_argument("--force", action="store_true", help="Force re-download/rebuild even if data exists.")

    # extract-features command
    parser_extract = subparsers.add_parser("extract_features", help="Extract all DSP features for the dataset.")
    parser_extract.add_argument("--overwrite", action="store_true", help="Overwrite existing feature caches.")
    parser_extract.add_argument("--output", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "features"), help="Directory to store extracted feature files.")

    # analyze command
    parser_analyze = subparsers.add_parser("analyze", help="Run temporal grounding and causal explanation on a given audio file.")
    parser_analyze.add_argument("audio_path", type=str, help="Path to the audio file to analyze.")
    parser_analyze.add_argument("--transcript", type=str, default=None, help="Optional transcript file (if not provided, Whisper will be used).")

    # evaluate command
    parser_eval = subparsers.add_parser("evaluate", help="Run full evaluation on the test split and generate reports.")
    parser_eval.add_argument("--output", type=str, default="results", help="Directory to store evaluation metrics and figures.")

    args = parser.parse_args()

    if args.command == "build_dataset":
        build_dataset(force=args.force)
    elif args.command == "extract_features":
        extract_features(MANIFEST_PATH, Path(args.output), overwrite=args.overwrite)
    elif args.command == "analyze":
        run_analysis(args.audio_path, transcript_path=args.transcript)
    elif args.command == "evaluate":
        run_evaluation(args.output)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
