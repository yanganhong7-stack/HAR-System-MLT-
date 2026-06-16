"""Run the K-Means experiment."""

from har_mlt.cli import main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--model", "kmeans", *sys.argv[1:]]
    main()
