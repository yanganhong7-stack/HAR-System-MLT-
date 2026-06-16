"""Run the Random Forest experiment."""

from har_mlt.cli import main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--model", "rf", *sys.argv[1:]]
    main()
