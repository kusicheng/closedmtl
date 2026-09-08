"""Command-line entry point for text translation."""

import argparse
import sys

from api_networking_components.translation_API import translation_request

def main():
    if len(sys.argv)>1 and sys.argv[1]=="images":
        from api_networking_components.image_pipeline import main as images_main
        return images_main(sys.argv[2:])
    parser=argparse.ArgumentParser(description="Detect a text's language and translate it through OpenRouter.")
    parser.add_argument("--text", help="Text to translate; reads standard input if omitted.")
    parser.add_argument("--target", required=True, help="Target language name or code, such as English or en.")
    parser.add_argument("--source", default="detect language", help="Source language; default: detect language.")
    parser.add_argument("--model", help="OpenRouter model ID; defaults to OPENROUTER_MODEL or the project default.")
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
