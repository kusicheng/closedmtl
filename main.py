"""Command-line entry point for text translation and the images subcommand."""

import argparse
import sys

from api_networking_components.translation_API import translation_request

def main():
    if len(sys.argv)>1 and sys.argv[1]=="images":
        from api_networking_components.image_pipeline import main as images_main
        return images_main(sys.argv[2:])
    parser=argparse.ArgumentParser(description="Detect a text's language and translate it")
    parser.add_argument("--text", help="Text to translate; if not provided, reads from stdin.")
    parser.add_argument("--target", required=True, help="Target language")
    parser.add_argument("--source", default="detect language", help="Source language; default: detect language.")
    parser.add_argument("--model", help="OpenRouter model ID; defaults to OPENROUTER_MODEL or minimax.")
    args=parser.parse_args()
    text=args.text if args.text is not None else sys.stdin.read()
    try:
        result=translation_request(text, target_language=args.target,
                                   source_language=args.source, model=args.model)
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(result)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
