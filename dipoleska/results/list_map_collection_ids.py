import contextlib
import os
from dipoleska.utils.map_read import MapCollectionLoader


def main() -> None:
    loader = MapCollectionLoader(use_base_rms=True)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        entries = loader.list_available()

    ids = sorted({entry["id"] for entry in entries if "id" in entry})
    if not ids:
        raise RuntimeError("No map collection IDs were discovered.")

    for collection_id in ids:
        print(collection_id)


if __name__ == "__main__":
    main()
